"""Reproduce the README cases with the published CLI solver and extra IPC relaxation.

Run from a Python 3.13 environment with untangling26==0.2 installed:
    python PythonBindings/examples/cli_relaxation.py --relax-steps 200
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh

from untangling26.cli import configure_solver, make_parser, select_backend
from untangling26.mesh_io import Body, load_bodies, normalize_bodies, validate_mesh, write_mesh

ROOT = Path(__file__).resolve().parents[2]


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def snapshot(solver):
    vertices, faces = solver.get_sim_result()
    return [np.array(v, copy=True) for v in vertices], [np.array(f, copy=True) for f in faces]


def export(path, bodies, mesh, center, scale):
    vertices, faces = mesh
    if len(vertices) != len(bodies):
        raise RuntimeError('Solver body count changed')
    result = []
    for body, v, f in zip(bodies, vertices, faces):
        v, f = validate_mesh(np.asarray(v, dtype=np.float64) * scale + center, f, body.name)
        result.append(Body(body.name, v, f, body.source))
    write_mesh(path, result)


def intersection_counts(solver):
    data = solver.get_intersection_contour_data()
    pairs = int(np.asarray(data['num_pairs']).reshape(-1)[0])
    contours = int(np.asarray(data['num_contours']).reshape(-1)[0])
    if min(pairs, contours) < 0:
        raise RuntimeError('Invalid collision report')
    return pairs, contours


def run_case(name, inputs, material, args):
    import lcs_py as lcs

    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=False)
    bodies = load_bodies(inputs, split=material == 'rigid')
    normalized, center, scale = normalize_bodies(bodies)
    cli_args = make_parser().parse_args([*[str(p) for p in inputs], str(folder / 'relaxed.obj'),
                                       '--material', material, '--backend', args.backend,
                                       '--physics', args.physics, '--max-iters', str(args.max_iters)])
    cli_args.gpu_untangling = args.gpu_untangling
    solver = lcs.NewtonSolver()
    records, relax_records = [], []
    started = time.perf_counter()
    manifest = {'case': name, 'package_version': version('untangling26'), 'material': material,
                'date': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
                'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                'rest_shape': 'Original input; retained in the same solver during relaxation',
                'normalization': {'center': center.tolist(), 'scale': scale},
                'untangling': records, 'relaxation': relax_records, 'status': 'running'}

    def save():
        (folder / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    try:
        backend = select_backend(lcs, args.backend)
        solver.init_device(backend_name=backend)
        for body in normalized:
            world = solver.create_world_data_from_array(body.name, body.vertices, body.faces)
            world.set_auto_scaling(False)
            if material == 'rigid':
                world.set_physics_material_rigid(thickness=1e-3, stiffness=1e4)
            else:
                world.set_physics_material_cloth()
            solver.register_world_data(world)
        solver.init_solver()
        config = solver.get_config()
        configure_solver(config, cli_args)
        manifest['backend'] = backend
        manifest['physics'] = args.physics
        manifest['gpu_untangling'] = args.gpu_untangling
        step = solver.physics_step_cpu if args.physics == 'cpu' else solver.physics_step_gpu
        export(folder / 'input.obj', bodies, snapshot(solver), center, scale)

        for iteration in range(args.max_iters + 1):
            checked_mesh = snapshot(solver)
            # Reports refer to the input of physics_step. Keep the full DOF
            # checkpoint, including affine rigid-body coordinates, for rollback.
            checkpoint = folder / 'checked.state'
            solver.save_current_state(str(checkpoint))
            step()
            pairs, contours = intersection_counts(solver)
            records.append({'iteration': iteration, 'ef_pairs': pairs, 'contours': contours})
            print(f'{name}: untangle {iteration}, EF={pairs}', flush=True)
            if pairs == 0 and contours == 0:
                solver.load_target_state(str(checkpoint))
                restored, _ = snapshot(solver)
                restore_error = max(float(np.max(np.abs(a - b))) for a, b in zip(restored, checked_mesh[0]))
                if restore_error > 2e-6:
                    raise RuntimeError(f'State restore error: {restore_error}')
                manifest['restore_max_normalized_error'] = restore_error
                manifest['untangling_updates'] = iteration
                export(folder / 'untangled.obj', bodies, checked_mesh, center, scale)
                break
        else:
            raise RuntimeError('Untangling budget exhausted; relaxation was not started')

        settings = dict(use_untangling=False, use_untangling_PRP=False,
                        use_untangling_ICM=False, use_untangling_GIA=False,
                        use_self_collision=True, use_ccd_linesearch=True, use_global_ccd=True,
                        contact_energy_type=1, use_energy_linesearch=True,
                        use_quasi_static_mode=True, nonlinear_iter_count=1)
        for key, value in settings.items():
            setattr(config, key, value)
        manifest['relaxation_settings'] = settings
        manifest['relaxation_steps_requested'] = args.relax_steps
        previous = snapshot(solver)[0]
        save()
        for iteration in range(1, args.relax_steps + 1):
            step()
            current = snapshot(solver)
            if not all(np.isfinite(v).all() for v in current[0]):
                raise RuntimeError('Non-finite coordinates during relaxation')
            displacement = max(float(np.max(np.linalg.norm(v - p, axis=1))) for v, p in zip(current[0], previous))
            relax_records.append({'iteration': iteration, 'max_normalized_vertex_step': displacement})
            previous = current[0]
            if iteration % 25 == 0 or iteration == args.relax_steps:
                export(folder / f'relax_{iteration:04d}.obj', bodies, current, center, scale)
                print(f'{name}: relax {iteration}/{args.relax_steps}, max step={displacement:.3g}', flush=True)
                save()

        # Restore the CLI's detector configuration for one final check. Its
        # speculative update is discarded; export only the matching snapshot.
        relaxed = snapshot(solver)
        configure_solver(config, cli_args)
        step()
        pairs, contours = intersection_counts(solver)
        export(folder / 'relaxed.obj', bodies, relaxed, center, scale)
        manifest.update(final_ef_pairs=pairs, final_contours=contours,
                        initial_ef_pairs=records[0]['ef_pairs'],
                        relaxation_steps_completed=len(relax_records),
                        elapsed_seconds=time.perf_counter() - started,
                        converged=pairs == 0 and contours == 0)
        if not manifest['converged']:
            raise RuntimeError(f'Relaxed output has {pairs} EF pairs / {contours} contours')
        manifest['status'] = 'complete'
        save()
        print(f'{name}: complete, {args.relax_steps} relaxation steps, EF=0', flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        save()
        raise
    finally:
        solver.cleanup_device()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'eight', 'klein', 'rigid'), default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/cli_relaxation')
    parser.add_argument('--relax-steps', type=nonnegative, default=200)
    parser.add_argument('--max-iters', type=nonnegative, default=200)
    parser.add_argument('--backend', default='auto', choices=('auto', 'vk', 'cuda', 'dx', 'metal'))
    parser.add_argument('--physics', default='cpu', choices=('cpu', 'gpu'))
    parser.add_argument('--gpu-untangling', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    cases = ('eight', 'klein', 'rigid') if args.case == 'all' else (args.case,)
    for name in cases:
        if (args.output / name).exists():
            parser.error(f'{args.output / name} exists; choose a new --output directory')
    klein = ROOT / 'external/instant-mesh-intersection-repair/data/misc/disc_kleinbottle.obj'
    if 'klein' in cases and not klein.exists():
        parser.error('Initialize the mesh submodule: git submodule update --init external/instant-mesh-intersection-repair')
    inputs = args.output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    if 'eight' in cases:
        shutil.copyfile(ROOT / 'Resources/InputMesh/SelfIntersectionUnit/unit_Eight.obj', inputs / 'eight.obj')
    if 'klein' in cases:
        shutil.copyfile(klein, inputs / 'klein.obj')
    if 'rigid' in cases:
        for name, translation in (('body_a', (10., 20., 30.)), ('body_b', (11.2, 20.4, 30.6))):
            box = trimesh.creation.box(extents=(2., 2., 2.))
            box.apply_translation(translation)
            box.export(inputs / (name + '.obj'))
    for name in cases:
        paths = [inputs / 'body_a.obj', inputs / 'body_b.obj'] if name == 'rigid' else [inputs / (name + '.obj')]
        run_case(name, paths, 'rigid' if name == 'rigid' else 'cloth', args)


if __name__ == '__main__':
    main()
