import matplotlib.pyplot as plt
import numpy as np

ANGLE_INC = np.pi / 6.


def print_bbox(bbox):
    x1, y1, x2, y2 = bbox
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
    return True


def bbox2para(bbox):
    x1, y1, x2, y2 = bbox
    return (x1, y1), (x2, y1), (x2, y2), (x1, y2)


def print_para(para, skip_outlier=False, width=None, height=None, color='r',
               circular=False, **kwargs):
    if skip_outlier:
        for x, y in para:
            if (x < 0 or (width and x >= width) or
                    y < 0 or (height and y >= height)):
                return False

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = para
    if circular:
        if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
            para = tuple((x+width, y) for x, y in para)
            print_para(
                para, skip_outlier, width, height, color, False, **kwargs)
        elif x1 >= width or x2 >= width or x3 >= width or x4 >= width:
            para = tuple((x-width, y) for x, y in para)
            print_para(
                para, skip_outlier, width, height, color, False, **kwargs)

    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color, **kwargs)
    return True


def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - (2*np.pi) * round(x / (2*np.pi))


def _adjust_elevation(h, e, delta_e):
    y = -np.sin(e)
    x = np.cos(e)*np.sin(h)
    z = np.cos(e)*np.cos(h)
    ee = -np.arctan2(y, z)
    hh = np.arcsin(x)

    ee += delta_e

    x = np.sin(hh)
    y = -np.cos(hh) * np.sin(ee)
    z = np.cos(hh) * np.cos(ee)
    h_new = np.arctan2(x, z)
    e_new = -np.arcsin(y)
    return h_new, e_new


def map_para2angles(para, viewIndex, width, height, depth):
    base_heading = (viewIndex % 12) * ANGLE_INC
    base_elevation = (viewIndex // 12 - 1) * ANGLE_INC

    angles = [None]*4
    for n, (X, Y) in enumerate(para):
        x = X - width/2.
        y = Y - height/2.
        z = depth
        r = np.sqrt(x**2 + y**2 + z**2)

        rel_heading = np.arctan2(x, z)
        rel_elevation = - np.arcsin(y / r)

        heading, elevation = _adjust_elevation(
            rel_heading, rel_elevation, base_elevation)
        heading = _canonical_angle(base_heading + heading)
        angles[n] = heading, elevation
    return tuple(angles)


def map_angles2para(angles, viewIndex, width, height, depth):
    base_heading = (viewIndex % 12) * ANGLE_INC
    base_elevation = (viewIndex // 12 - 1) * ANGLE_INC

    paras = [None]*4
    for n, (heading, elevation) in enumerate(angles):
        heading = _canonical_angle(heading - base_heading)
        rel_heading, rel_elevation = _adjust_elevation(
            heading, elevation, -base_elevation)
        # Check whether the point is in the half-sphere (visible)
        if rel_heading <= -np.pi/2 or rel_heading > np.pi/2:
            return None, False

        z = depth
        x = z * np.tan(rel_heading)
        y = -np.tan(rel_elevation) * np.sqrt(x**2 + z**2)
        X = x + width/2.
        Y = y + height/2.
        paras[n] = X, Y
    return tuple(paras), True


def gen_panorama_img(scanId, viewpointId, agent_heading=0, vfov_deg=90,
                     hfov_deg=10, height=480):
    # Try to import the matterport simulator
    import os.path as osp
    import sys; sys.path.append(osp.join(osp.dirname(__file__), '../../../build'))  # NoQA
    import MatterSim

    assert 360 % hfov_deg == 0, 'hfov_deg must be a factor of 360'
    HEIGHT = height
    VFOV = np.radians(vfov_deg)
    HFOV = np.radians(hfov_deg)
    DEPTH = (HEIGHT/2.) / np.tan(VFOV/2.)
    WIDTH = int(np.round(2.*DEPTH*np.tan(HFOV/2.)))

    num_headings = int(360/hfov_deg) + 1
    pano_img = np.zeros(
        (HEIGHT, WIDTH*num_headings, 3), np.uint8)

    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.init()
    # sim.initialize()
    for n_heading in range(num_headings):
        heading = agent_heading+np.radians(n_heading*hfov_deg-180)
        sim.newEpisode(scanId, viewpointId, heading, 0)
        state = sim.getState()
        im = state.rgb
        X_begin = WIDTH*n_heading
        X_end = X_begin+WIDTH
        pano_img[:, X_begin:X_end, :] = im[..., ::-1]
    pano_img = pano_img[:, WIDTH//2:-WIDTH//2, :]
    return pano_img


def rotate_panorama_img(img, new_agent_heading, old_agent_heading=0):
    img_new = np.zeros_like(img)
    rotation = _canonical_angle(new_agent_heading - old_agent_heading)
    split_X = int(np.round(img.shape[1] * rotation / (2*np.pi)))
    if split_X == 0:
        img_new[...] = img
    else:
        img_new[:, -split_X:, :] = img[:, :split_X, :]
        img_new[:, :-split_X, :] = img[:, split_X:, :]
    return img_new


def map_angles_onto_panorama(
        angles, width, height, agent_heading=0, vfov_deg=90):
    VFOV = np.radians(vfov_deg)
    depth = (height/2.) / np.tan(VFOV/2.)

    paras = [None]*4

    base_heading = max(_canonical_angle(a[0] - agent_heading) for a in angles)
    base_X = width * (base_heading + np.pi) / (2*np.pi)
    for n, (heading, elevation) in enumerate(angles):
        X = base_X + width * _canonical_angle(
                heading - agent_heading - base_heading) / (np.pi*2)
        Y = height/2. - depth * np.tan(elevation)
        paras[n] = X, Y

    return paras
