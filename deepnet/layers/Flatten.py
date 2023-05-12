import numpy as np


class Flatten:
    def matrices2vector(self, matrices):
        vector = np.array([[]], dtype=np.float32)
        self.matrix_shape = matrices[0].shape
        for i in range(len(matrices)):
            reshaped_matrix = np.reshape(
                matrices[i], (1, self.matrix_shape[0] * self.matrix_shape[1]))
            vector = np.hstack((vector, reshaped_matrix))
        return vector

    def vector2matrices(self, vector):
        matrices = []
        matrix_size = self.matrix_shape[0] * self.matrix_shape[1]
        for i in range(0, vector.size, matrix_size):
            matrix = np.reshape(vector[0][i:i + matrix_size], self.matrix_shape)
            matrices.append(matrix)
        return matrices
