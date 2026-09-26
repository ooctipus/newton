"""Current-main source admission and untimed observations for the existing harness."""
import hashlib
import ast
import json
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent / 'helpers'
sys.path.insert(0, str(TOOLS))
import checked_capture as checked
import compare_backends as owner


def main():
    expected = json.loads(Path(os.environ['SPARSE_SOURCE_ADMISSION']).read_text())
    capture = owner.load_capture(Path(expected['isaaclab']))
    kernel_overrides = {'sparse_mass_matrix': False}
    hook_args = [sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1])
                 if value == '--solver-attr' and sys.argv[i + 1].startswith('_kernel_overrides=')]
    if len(hook_args) != 1:
        raise RuntimeError('Specify exactly one explicit sparse kernel control')
    kernel_overrides = ast.literal_eval(hook_args[0].split('=', 1)[1])
    if (not isinstance(kernel_overrides, dict) or set(kernel_overrides) != {'sparse_mass_matrix'}
            or type(kernel_overrides['sparse_mass_matrix']) is not bool):
        raise ValueError('Only the explicit boolean sparse_mass_matrix test hook is admitted')

    def admit(root):
        if str(root) != expected['root'] or capture._source(root) != expected['source']:
            raise RuntimeError('Capture requires the exact selected Newton commit and frozen source delta')

    original_select = checked.select_imports

    def select(lab, newton):
        selected = original_select(lab, newton)
        from newton.solvers import SolverFeatherPGS
        if kernel_overrides['sparse_mass_matrix'] and not hasattr(SolverFeatherPGS, '_setup_sparse_mass_matrix'):
            raise RuntimeError('Selected source does not implement the sparse test hook')
        # This existing hook is a class variable, NOT a constructor argument;
        # merely placing it in Lab solver_cfg would otherwise silently do nothing.
        SolverFeatherPGS._kernel_overrides = dict(kernel_overrides)
        return selected

    original_row = checked.legacy_row_status

    def observe(solver, entry):
        original_row(solver, entry)
        solver.check_constraint_capacity()
        entry['current_public_constraint_check'] = True
        entry['execution'] = {name: getattr(solver, name, None) for name in (
            'parallel_tree', 'articulated_contact_response', 'use_parallel_streams',
            '_sparse_diagonal_contact_solve', '_paired_response_primary_size',
            '_paired_response_secondary_size', '_sparse_diagonal_response_size',
            '_compact_diagonal_mass_size', '_row_watermark', 'enable_contact_friction',
            'enable_joint_limits', 'enable_joint_velocity_limits',
            'update_mass_matrix_interval', 'pgs_iterations', 'pgs_inner_substeps',
            'pgs_velocity_iterations')}
        entry['constraint_row_watermarks'] = solver.constraint_row_watermarks()
        entry['watermarks_enabled'] = bool(solver._row_watermark)
        entry['kernel_overrides'] = dict(solver._kernel_overrides)
        entry['sparse_mass_matrix_size'] = getattr(solver, '_sparse_mass_matrix_size', None)
        entry['parallel_tree_active'] = getattr(solver, '_tree_plan', None) is not None
        entry['response_storage_shapes'] = {
            name: {str(size): list(array.shape) for size, array in getattr(solver, name, {}).items()
                   if array is not None}
            for name in ('H_by_size', 'L_by_size', 'J_by_size', 'Y_by_size', 'diag_by_size')}
        if entry['sparse_mass_matrix_size'] is not None:
            status = solver._sparse_mass_matrix_status.numpy()
            entry['sparse_representation'] = dict(
                dofs=solver._sparse_mass_matrix_plan.dof_count,
                packed_factor_nnz=solver._sparse_mass_matrix_plan.nonzero_count,
                max_row_support=solver._sparse_row_dof.shape[-1],
                factor_warps_per_block=solver._sparse_factor_warps_per_block,
                factor_status_nonzero=int((status != 0).sum()),
                factor_status_max=int(status.max()),
                factor_kernel=solver._crba_sparse_factor_kernel.key,
                solve_kernel=solver._pgs_solve_sparse_kernel.key,
                selected_contact_producer='populate_sparse_contact_response',
                selected_joint_limit_producer='build_sparse_joint_limit_rows',
                dispatch_evidence='Selected runtime representation and source branch; not per-kernel tracing',
                double_buffer=solver._double_buffer)
            if (status != 0).any():
                raise RuntimeError('Sparse factorization status is nonzero')

    original_install = checked.install_boundary_check

    def install(harness, report, *args, **kwargs):
        report['current_main_source_admission'] = expected
        report['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['requested_kernel_overrides'] = kernel_overrides
        report['scope'] = ('Current public constraint check plus retained row warning flags and '
                           'collision boundary snapshots/full-log warning rejection; not a no-drop certificate.')
        return original_install(harness, report, *args, **kwargs)

    checked.validate_legacy_source = admit
    checked.select_imports = select
    checked.legacy_row_status = observe
    checked.install_boundary_check = install
    return checked.main()


if __name__ == '__main__':
    raise SystemExit(main())
