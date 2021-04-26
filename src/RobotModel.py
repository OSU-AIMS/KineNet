import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class RobotModel:

    def __init__(self, urdf):
        # Rotational axis is character format +/- x/y/z
        self._model = pd.DataFrame(
            columns=['x', 'y', 'z', 'rot-axis', 'angle-min', 'angle-max', 'length', 'parent', 'child'])
        self._model.loc['base'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '', '']
        self._file = ET.parse(urdf)
        self._root = self._file.getroot()
        self.__parse_tree()

    def show(self):
        print(self._model)

    def joint_idx(self):
        idx = self._model[['x', 'y', 'z']]
        return idx

    def rotation_axis(self):
        rot_axis = self._model['rot-axis']
        return rot_axis

    def phys_limits(self):
        limits = self._model[['length', 'angle-min', 'angle-max']]
        return limits

    def affix_base_joint(self, train_batch):
        tr = np.empty((train_batch.shape[0], train_batch.shape[1]+1, train_batch.shape[2]))
        idx, ind = self.joint_idx(), self._model.index
        dbase = -1.0 * idx.loc[ind[1]].to_numpy()
        for R in range(train_batch.shape[0]):
            tr[R][:] = np.insert(train_batch[R][:], 0, [train_batch[R][0] + dbase], axis=0)
        return tr

    @staticmethod
    def __convert_aor(axis):
        aor = ''
        for i in range(len(axis)):
            if axis[i] != 0.0:
                if i % 3 == 0:
                    aor += 'x'
                elif i % 3 == 1:
                    aor += 'y'
                else:
                    aor += 'z'
                if axis[i] < 0.0:
                    aor = '-' + aor
        return aor

    def node_dist(self, C, P):
        C, P = self._model.loc[C][['x', 'y', 'z']], self._model.loc[P][['x', 'y', 'z']]
        return np.sqrt(np.square(C['x'] - P['x']) + np.square(C['y'] - P['y']) + np.square(C['z'] - P['z']))

    def __fix_position(self, name, parent):
        if parent != 'base':
            P = self._model.loc[parent][['x', 'y', 'z']]
            self._model.at[name, 'x'] += P['x']
            self._model.at[name, 'y'] += P['y']
            self._model.at[name, 'z'] += P['z']
            parent = self._model.loc[parent]['parent']
            self.__fix_position(name, parent)

    def __parse_tree(self):
        to_float = lambda x: [float(i) for i in x.split()]
        parent = 'base'
        for joint in self._root.findall('joint'):
            if joint.attrib['type'] == 'revolute':
                name = joint.find('child').attrib['link']
                self._model.at[parent, 'child'] = name
                xyz = to_float(joint.find('origin').attrib['xyz'])
                aor = RobotModel.__convert_aor(to_float(joint.find('axis').attrib['xyz']))
                low, high = float(joint.find('limit').attrib['lower']), float(joint.find('limit').attrib['upper'])
                self._model.at[name, :] = [xyz[0], xyz[1], xyz[2], aor, low, high, 0.0, parent, '']
                self.__fix_position(name, parent)
                self._model.at[name, 'length'] = self.node_dist(name, parent)
                parent = name