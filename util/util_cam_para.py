import numpy as np


def read_cam_para_from_xml(xml_name):
    # azi ele only
    import xml.etree.ElementTree
    e = xml.etree.ElementTree.parse(xml_name).getroot()

    assert len(e.findall('sensor')) == 1
    for x in e.findall('sensor'):
        assert len(x.findall('transform')) == 1
        for y in x.findall('transform'):
            assert len(y.findall('lookAt')) == 1
            for z in y.findall('lookAt'):
                origin = np.array(z.get('origin').split(','), dtype=np.float32)
                # up = np.array(z.get('up').split(','), dtype=np.float32)

    x, y, z = origin
    elevation = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))
    azimuth = np.arctan2(x, z) + np.pi
    if azimuth >= np.pi:
        azimuth -= 2 * np.pi
    assert azimuth >= -np.pi and azimuth <= np.pi
    assert elevation >= -np.pi / 2. and elevation <= np.pi / 2.
    return azimuth, elevation


def raw_camparam_from_xml(path, pose="lookAt"):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    elm = tree.find("./sensor/transform/" + pose)
    camparam = elm.attrib
    origin = np.fromstring(camparam['origin'], dtype=np.float32, sep=',')
    target = np.fromstring(camparam['target'], dtype=np.float32, sep=',')
    up = np.fromstring(camparam['up'], dtype=np.float32, sep=',')
    height = int(
        tree.find("./sensor/film/integer[@name='height']").attrib['value'])
    width = int(
        tree.find("./sensor/film/integer[@name='width']").attrib['value'])

    camparam = dict()
    camparam['origin'] = origin
    camparam['up'] = up
    camparam['target'] = target
    camparam['height'] = height
    camparam['width'] = width
    return camparam


def get_object_rotation(xml_path, style='zup'):
    style_set = ['yup', 'zup', 'spherical_proj']
    assert(style in style_set)
    camparam = raw_camparam_from_xml(xml_path)
    if style == 'zup':
        Rx = camparam['target'] - camparam['origin']
        up = camparam['up']
        Rz = np.cross(Rx, up)
        Ry = np.cross(Rz, Rx)
        Rx /= np.linalg.norm(Rx)
        Ry /= np.linalg.norm(Ry)
        Rz /= np.linalg.norm(Rz)
        R = np.array([Rx, Ry, Rz])
        R_coord = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        R = R_coord @ R
        R = R @ R_coord.transpose()
    elif style == 'yup':
        Rx = camparam['target'] - camparam['origin']
        up = camparam['up']
        Rz = np.cross(Rx, up)
        Ry = np.cross(Rz, Rx)
        Rx /= np.linalg.norm(Rx)
        Ry /= np.linalg.norm(Ry)
        Rz /= np.linalg.norm(Rz)
        #print(Rx, Ry, Rz)
        # no transpose needed!
        R = np.array([Rx, Ry, Rz])
    elif style == 'spherical_proj':
        Rx = camparam['target'] - camparam['origin']
        up = camparam['up']
        Rz = np.cross(Rx, up)
        Ry = np.cross(Rz, Rx)
        Rx /= np.linalg.norm(Rx)
        Ry /= np.linalg.norm(Ry)
        Rz /= np.linalg.norm(Rz)
        #print(Rx, Ry, Rz)
        # no transpose needed!
        R = np.array([Rx, Ry, Rz])

        raise NotImplementedError
    return R


def get_object_rotation_translation(xml_path, style='zup'):
    pass


def _devide_into_section(angle, num_section, lower_bound, upper_bound):
    rst = np.zeros(num_section)
    per_section_size = (upper_bound - lower_bound) / num_section
    angle -= per_section_size / 2
    if angle < lower_bound:
        angle += upper_bound - lower_bound
    idx = int((angle - lower_bound) / per_section_size)
    rst[idx] = 1
    return rst


def _section_to_angle(idx, num_section, lower_bound, upper_bound):
    per_section_size = (upper_bound - lower_bound) / num_section

    angle = (idx + 0.5) * per_section_size + lower_bound
    angle += per_section_size / 2
    if angle > upper_bound:
        angle -= upper_bound - lower_bound
    return angle


def azimuth_to_onehot(azimuth, num_azimuth):
    return _devide_into_section(azimuth, num_azimuth, -np.pi, np.pi)


def elevation_to_onehot(elevation, num_elevation):
    return _devide_into_section(elevation, num_elevation, -np.pi / 2., np.pi / 2.)


def onehot_to_azimuth(v, num_azimuth):
    idx = np.argmax(v)
    return _section_to_angle(idx, num_azimuth, -np.pi, np.pi)


def onehot_to_elevation(v, num_elevation):
    idx = np.argmax(v)
    return _section_to_angle(idx, num_elevation, -np.pi / 2., np.pi / 2.)


if __name__ == '__main__':
    num_azimuth = 24
    num_elevation = 12
    for i in range(num_azimuth):
        rst = np.zeros(num_azimuth)
        rst[i] = 1
        print(onehot_to_azimuth(rst, num_azimuth))

    '''
    for i in range(100):
        angle = (np.random.rand() - 0.5) * np.pi * 2
        print(angle, np.argmax(azimuth_to_onehot(angle, 24)), onehot_to_azimuth(azimuth_to_onehot(angle, 24), 24))
        assert np.abs(angle - onehot_to_azimuth(azimuth_to_onehot(angle, 24), 24)) < 2 * np.pi / 24
    '''
