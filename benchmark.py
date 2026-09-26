"""Reuse the existing task/capture harness at 16K, selected clean Newton, RTX only."""
import argparse
import ast
import json
from pathlib import Path
import signal
import sys

from capture_adapter import TOOLS
sys.path.insert(0, str(TOOLS))
import compare_backends as owner

ROOT = Path(__file__).resolve().parent
BASE = '5238407d320823e71a5a4623c2c83c293d7b00fd'
ORDER = ['franka', 'kuka', 'allegro', 'anymald', 'g1', 'keyboard-so101', 'ant',
         'anymald-rough', 'cartpole', 'cassie-flat', 'cassie-rough', 'franka-reach',
         'g1-flat', 'go2-flat', 'go2-rough', 'h1-flat', 'h1-rough', 'humanoid', 'ur10-reach', 'kuka-reorient']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--newton', type=Path, required=True)
    parser.add_argument('--isaaclab', type=Path, required=True)
    parser.add_argument('--capacity-file', type=Path, default=ROOT / 'capacities.json')
    parser.add_argument('--gpu-uuid', required=True, help='Explicitly admit the selected GPU0 UUID')
    parser.add_argument('--commit', default=BASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--allow-dirty-candidate', action='store_true')
    parser.add_argument('--solver-attr', action='append', default=[])
    parser.add_argument('--num-envs', type=int, default=16384)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--profile-steps', type=int, default=40)
    args = parser.parse_args()
    LAB = args.isaaclab.resolve()
    PRIOR = args.capacity_file.resolve()
    args.isaaclab = LAB
    args.task = args.task or ORDER
    if min(args.num_envs, args.steps, args.profile_steps) < 1 or args.warmup_steps < 0:
        parser.error('Positive sample sizes and nonnegative warmup required')
    args.check_overflow, args.mjwarp_linesearch_fix = True, False
    if any(task not in ORDER for task in args.task):
        parser.error('Select canonical non-Drawer tasks')
    capture = owner.load_capture(LAB)
    capture.RECIPES['kuka-reorient'] = ('Isaac-Reorient-KukaAllegro', (), {})
    # Strip every historical research opt-in, preserving only supported explicit solver attributes.
    tree = ast.parse((args.newton / 'newton/_src/solvers/feather_pgs/solver_feather_pgs.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SolverFeatherPGS')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    supported = {a.arg for a in [*init.args.args, *init.args.kwonlyargs]}
    if any(v.partition('=')[0] not in supported | {'_kernel_overrides'} for v in args.solver_attr):
        parser.error('Explicit solver attribute is not supported by the selected source')
    if not any(v.startswith('_kernel_overrides=') for v in args.solver_attr):
        args.solver_attr.append("_kernel_overrides={'sparse_mass_matrix': False}")
    stripped = {}
    for name, (task, attrs, flags) in capture.RECIPES.items():
        kept = tuple(v for v in attrs if v.partition('=')[0] in supported)
        stripped[name] = dict(attributes=[v for v in attrs if v not in kept], environment=flags)
        capture.RECIPES[name] = task, kept, {}
    stripped['common'] = {k: v for k, v in capture.COMMON_ENVIRONMENT.items() if k.startswith(owner.FLAG_PREFIXES)}
    capture.COMMON_ENVIRONMENT = {k: v for k, v in capture.COMMON_ENVIRONMENT.items() if not k.startswith(owner.FLAG_PREFIXES)}
    prior = json.loads(PRIOR.read_text())
    args.capacity = []
    for task in args.task:
        source_task = 'kuka' if task == 'kuka-reorient' else task
        for item in prior['tasks'].get(source_task, {}).get('capacity_overrides', []):
            key, value = item.split('=')
            key = key.replace(source_task + ':', task + ':', 1)
            if ':fpgs:' not in key:
                continue
            # Global storage scales with world count; per-world rows do not.
            value = int(value)
            if key.split(':')[-1] in owner.COLLISION_CAPACITIES:
                value = (value * args.num_envs + 4095) // 4096
            args.capacity.append(f'{key}={value}')
    roots = dict(isaaclab=LAB, newton=args.newton.resolve())
    sources = {name: capture._source(path) for name, path in roots.items()}
    if args.allow_dirty_candidate and args.commit == BASE:
        parser.error('Immutable baseline cannot admit dirty state')
    if (sources['newton']['sha'] != args.commit
            or (sources['newton']['dirty'] and not args.allow_dirty_candidate)):
        raise RuntimeError('Require selected commit, and explicit admission for a frozen dirty candidate')
    if sources['isaaclab']['sha'] != '53ee6b44c2334341305dbdf385a3916c6b140799':
        raise RuntimeError('Lab pin changed')
    files = owner.file_hashes([Path(__file__), Path(__file__).with_name('capture_adapter.py'),
        Path(__file__).with_name('launch_capture.py'), PRIOR, Path(owner.__file__), Path(capture.__file__),
        *(TOOLS / n for n in ('checked_capture.py', 'nsys_checked.sh')),
        *(capture.HARNESS / n for n in ('run_profiled.py', 'analyze_nsys.py'))])
    plan = dict(tasks=args.task, sampling=dict(num_envs=args.num_envs, warmup_steps=args.warmup_steps,
        steps=args.steps, profile_steps=args.profile_steps, repeats=1, seed=0, video=False), capacities=args.capacity,
        explicit_solver_attributes=args.solver_attr, allow_dirty_candidate=args.allow_dirty_candidate,
        capacity_scope='Historical 4K global storage scaled to worlds; unchanged per-world row storage. Not fresh demand calibration.',
        stripped_research_options=stripped, sources=sources, drivers=files)
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    owner.validate_output(args.output, [*roots.values(), TOOLS])
    args.output.mkdir(parents=True)
    admission = args.output / 'source-admission.json'
    capture._write_json(admission, dict(root=str(args.newton.resolve()), commit=args.commit,
        isaaclab=str(LAB), source=sources['newton'], allow_dirty_candidate=args.allow_dirty_candidate))
    files.update(owner.file_hashes([admission]))
    devices = capture._gpus([0])
    if devices[0]['uuid'] != args.gpu_uuid:
        raise RuntimeError('GPU0 UUID does not match explicit admission')
    manifest = dict(status='running', plan=plan, gpus=devices,
                    runtime=owner.runtime_software(LAB, {'selected': args.newton}), tasks={})
    def save():
        capture._write_json(args.output / 'manifest.json', manifest)
    save()
    for task in args.task:
        entry = manifest['tasks'][task] = dict(status='running', runs=[])
        try:
            capture._require_idle(devices)
            owner.source_guard(capture, roots, sources, files)
            directory = args.output / task
            directory.mkdir()
            runs = owner.make_batch(capture, args, {'fpgs': args.newton}, {0: {}}, devices,
                                    directory, 0, task, 'fpgs', drivers=files)
            entry['runs'] = runs
            for run in runs:
                run['environment'].update(FPGS_BENCH_LEGACY_CAPACITY='1', SPARSE_SOURCE_ADMISSION=str(admission))
                for attribute in args.solver_attr:
                    run['command'].extend(['--solver-attr', attribute])
                run['command'] = ['timeout', '120s' if task == 'franka' else '900s', sys.executable,
                                  str(Path(__file__).with_name('launch_capture.py')), *run['command'][2:]]
            save()
            owner.run_batch(runs, LAB)
            for run in runs:
                owner.check_overflow_result(run, files)
                run['result'] = capture._read_result(run)
                run['artifacts'] = owner.file_hashes([Path(run['output_dir']) / n for n in
                    ('capture.json', 'capture_checks.json', 'capture_analysis.json', 'capture.log')])
            entry.update(status='complete', summary=capture._summaries(runs))
        except Exception as exc:
            entry.update(status='failed', error=repr(exc))
        finally:
            save()
            owner.source_guard(capture, roots, sources, files)
            capture._require_idle(devices)
        print(task, entry['status'], entry.get('summary', entry.get('error')), flush=True)
    manifest['status'] = 'complete_with_failures' if any(v['status'] == 'failed' for v in manifest['tasks'].values()) else 'complete'
    save()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, owner.interrupted)
    main()
