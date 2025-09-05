import os
import numpy as nm

from sfepy.base.base import Output
from sfepy.discrete.fem import MeshIO
from sfepy.linalg import get_coors_in_ball
from sfepy import data_dir

output = Output('balloon')

def get_nodes(coors, radius, eps, mode):
    if mode == 'ax1':
        centre = nm.array([0.0, 0.0, -radius], dtype=nm.float64)
    elif mode == 'ax2':
        centre = nm.array([0.0, 0.0, radius], dtype=nm.float64)
    elif mode == 'equator':
        centre = nm.array([radius, 0.0, 0.0], dtype=nm.float64)
    else:
        raise ValueError('unknown mode %s!' % mode)

    return get_coors_in_ball(coors, centre, eps)

def get_volume(ts, coors, region=None, **kwargs):
    rs = 1.0 + 1.0 * ts.time
    rv = rs**3
    out = nm.empty((coors.shape[0],), dtype=nm.float64)
    out.fill(rv)
    return out

def define(plot=False, use_lcbcs=True):
    filename_mesh = data_dir + '/meshes/3d/unit_ball.mesh'
    conf_dir = os.path.dirname(__file__)
    io = MeshIO.any_from_filename(filename_mesh, prefix_dir=conf_dir)
    bbox = io.read_bounding_box()
    dd = bbox[1] - bbox[0]
    radius = bbox[1, 0]
    eps = 1e-8 * dd[0]

    options = {
        'nls' : 'newton',
        'ls' : 'ls',
        'ts' : 'ts',
        'save_times' : 'all',
        'output_dir' : '.',
        'output_format' : 'vtk',
    }

    fields = {
        'displacement': (nm.float64, 3, 'Omega', 1),
    }
    if use_lcbcs:
        fields['pressure'] = (nm.float64, 1, 'Omega', 0)
    else:
        fields['pressure'] = (nm.float64, 1, 'Omega', 0, 'L2', 'constant')

    # Updated: higher thickness and nonzero kappa.
    materials = {
        'solid' : ({
            'mu' : 50,
            'kappa' : 1e3,    # nonzero => compressible
        },),
        'walls' : ({
            'mu' : 3e5,
            'kappa' : 1e4,    # nonzero => compressible
            'h0' : 0.1,       # increased thickness
        },),
    }

    variables = {
        'u' : ('unknown field', 'displacement', 0),
        'v' : ('test field', 'displacement', 'u'),
        'p' : ('unknown field', 'pressure', 1),
        'q' : ('test field', 'pressure', 'p'),
        'omega' : ('parameter field', 'pressure', {'setter' : 'get_volume'}),
    }

    regions = {
        'Omega'  : 'all',
        'Ax1' : ('vertices by get_ax1', 'vertex'),
        'Ax2' : ('vertices by get_ax2', 'vertex'),
        'Equator' : ('vertices by get_equator', 'vertex'),
        'Surface' : ('vertices of surface', 'facet'),
    }

    ebcs = {
        'fix1' : ('Ax1', {'u.all' : 0.0}),
        'fix2' : ('Ax2', {'u.[0, 1]' : 0.0}),
        'fix3' : ('Equator', {'u.1' : 0.0}),
    }

    if use_lcbcs:
        lcbcs = {
            'pressure' : ('Omega', {'p.all' : None},
                          None, 'integral_mean_value'),
        }

    equations = {
        'balance' : """
          dw_tl_he_neohook.2.Omega(solid.mu, v, u)
        + dw_tl_he_mooney_rivlin.2.Omega(solid.kappa, v, u)
        + dw_tl_membrane.2.Surface(walls.mu, walls.kappa, walls.h0, v, u)
        + dw_tl_bulk_pressure.2.Omega(v, u, p)
        = 0""",
        'volume' : """
          dw_tl_volume.2.Omega(q, u)
        = dw_dot.2.Omega(q, omega)""",
    }

    solvers = {
        'ls' : ('ls.auto_direct', {
            'use_presolve' : False,
            'use_mtx_digest' : False,
        }),
        'newton' : ('nls.newton', {
            'i_max'      : 8,
            'eps_a'      : 1e-4,
            'eps_r'      : 1e-8,
            'lin_red'    : None,
            'ls_red'     : 0.5,
            'ls_red_warp': 0.1,
            'ls_on'      : 100.0,
            'ls_min'     : 1e-5,
            'check'      : 0,
            'delta'      : 1e-6,
            'report_status' : True,
        }),
        'ts' : ('ts.adaptive', {
            't0' : 0.0,
            't1' : 5.0,
            'dt' : None,
            'n_step' : 11,
            'dt_red_factor' : 0.8,
            'dt_red_max' : 1e-3,
            'dt_inc_factor' : 1.25,
            'dt_inc_on_iter' : 4,
            'dt_inc_wait' : 3,
            'verbose' : 1,
            'quasistatic' : True,
        }),
    }

    functions = {
        'get_ax1' : (lambda coors, domain:
                     get_nodes(coors, radius, eps, 'ax1'),),
        'get_ax2' : (lambda coors, domain:
                     get_nodes(coors, radius, eps, 'ax2'),),
        'get_equator' : (lambda coors, domain:
                         get_nodes(coors, radius, eps, 'equator'),),
        'get_volume' : (get_volume,),
    }

    return locals()
