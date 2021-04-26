import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class RobotModel:
    """
    URDF Parser class.
    :var self._model: Dataframe indexed by link name for each link. Contains <x,y,z> relative location in initial state,
                        angle max/min, link length from ancestor, and parent/child link names.
    :type self._model: pandas dataframe
    """

    def __init__(self, urdf):
        """
        Default constructor
        :param urdf: file location of robot urdf file
        :type urdf: string
        """
        # Rotational axis is character format +/- x/y/z
        self._model = pd.DataFrame(
            columns=['x', 'y', 'z', 'rot-axis', 'angle-min', 'angle-max', 'length', 'parent', 'child'])
        self._model.loc['base'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '', '']
        self._file = ET.parse(urdf)
        self._root = self._file.getroot()
        self.__parse_tree()

    def show(self):
        """
        Prints current model of robot
        """
        print(self._model)

    def joint_idx(self):
        """
        Accesses the distance vector between each revolute joint in the description file
        :rtype: numpy matrix - dims = [7 x 3]
        """
        idx = self._model[['x', 'y', 'z']]
        return idx

    def rotation_axis(self):
        """
        Gets URDF designated direction of rotation for each joint angle
        :rtype: numpy matrix - dims = [7 x 1]
        """
        rot_axis = self._model['rot-axis']
        return rot_axis

    def phys_limits(self):
        """
        Fetch link length between two joints relative to its ancestor
            e.g. link_t_limit = dist(link_t_coords - link_b_coords)
        Additionally fetches min and max rotational angle for each joint
        :rtype: numpy matrix - dims = [7 x 3]
        """
        limits = self._model[['length', 'angle-min', 'angle-max']]
        return limits

    def affix_base_joint(self, train_batch):
        """
        Appends base joint position to an existing numpy array
        :param train_batch: Matrix containing a series of numpy matrices of each joint coordinate vector
        :type train_batch: numpy matrix - dims = [(samples) x (joints) x 3]
        :return: adjusted numpy matrix with base coordinate appended for all entries
        :rtype: numpy matrix - dims = [(samples) x (joints) + 1 x 3]
        """
        tr = np.empty((train_batch.shape[0], train_batch.shape[1]+1, train_batch.shape[2]))
        idx, ind = self.joint_idx(), self._model.index
        dbase = -1.0 * idx.loc[ind[1]].to_numpy()
        for R in range(train_batch.shape[0]):
            tr[R][:] = np.insert(train_batch[R][:], 0, [train_batch[R][0] + dbase], axis=0)
        return tr

    @staticmethod
    def __convert_aor(axis):
        """
        Converts the axis of rotation from a one-hot vector to a single row with direction of rotation
        in x,y,z with direction specified by +/-
        :param axis: one-hot vector with entries [-1,1] to specify direction and axis of rotation
        :type axis: list [1 x 3]
        :return: +/- axis of rotation
        :rtype: string
        """
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
        """
        Distance function between to calculate link length
        :param C: name of child link
        :type C: string
        :param P: name of parent link
        :type P: string
        :return: link length between two vectors
        :rtype: double
        """
        C, P = self._model.loc[C][['x', 'y', 'z']], self._model.loc[P][['x', 'y', 'z']]
        return np.sqrt(np.square(C['x'] - P['x']) + np.square(C['y'] - P['y']) + np.square(C['z'] - P['z']))

    def __fix_position(self, name, parent):
        """
        Adjusts each joint to build initial state of robot such that true <x,y,z> for each joint relative
        to ancestors is formulated. Recursively updates self._model to make updates.
        :param name: link name
        :type name: string
        :param parent: parent link name
        :type parent: string
        """
        if parent != 'base':
            P = self._model.loc[parent][['x', 'y', 'z']]
            self._model.at[name, 'x'] += P['x']
            self._model.at[name, 'y'] += P['y']
            self._model.at[name, 'z'] += P['z']
            parent = self._model.loc[parent]['parent']
            self.__fix_position(name, parent)

    def __parse_tree(self):
        """
        Parses URDF xml tree.
        """
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