import copy
import time
import csv

import numpy as np

class Sudoku:
    def __init__(self):
        self.grid = np.zeros((9, 9)).astype(int)
        self.grid_config = np.zeros((9, 9)).astype(int)
        self.violations = np.zeros((9, 9)).astype(int)
        self.square_violations = np.zeros(9).astype(int)
        self.read_sudoku()
        print('Initial grid:\n')
        self.print_sudoku()
        self.solved = 0

    def read_sudoku(self, name="grid"):
        with open(name + ".csv", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                row = [cell.strip() for cell in row if cell.strip() != ""]
                for j, cell in enumerate(row):
                    try:
                        value = int(cell)
                    except ValueError:
                        value = 0
                    self.grid[i, j] = value
                    if value != 0:
                        self.grid_config[i, j] = 1

    def print_sudoku(self):
        for i in range(len(self.grid)):
            line = ""
            if i == 3 or i == 6:
                print("---------------------")
            for j in range(len(self.grid[i])):
                if j == 3 or j == 6:
                    line += "| "
                if self.grid_config[i, j] == 1:
                    line += "\033[1m" + str(self.grid[i][j]) + "\033[0m" + " "
                else:
                    value = '.' if self.grid[i][j] == 0 else str(self.grid[i][j])
                    line += value + " "
            print(line)
        print("\n")

    def fill_sudoku(self):
        for i in range(9):
            p = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9])
            for j in range(9):
                p = np.delete(p, np.where(p == self.grid[i // 3 * 3 + j // 3, j % 3 + (3 * i) % 9]))
            k = 0
            for j in range(9):
                if self.grid_config[i // 3 * 3 + j // 3, j % 3 + (3 * i) % 9] == 0:
                    self.grid[i // 3 * 3 + j // 3, j % 3 + (3 * i) % 9] = p[k]
                    if len(p) == 1:
                        self.grid_config[i // 3 * 3 + j // 3, j % 3 + (3 * i) % 9] = 1
                    k += 1

    def likelihood(self, sudoku):
        energy = 0
        for i in range(9):
            for j in range(9):
                value = sudoku[i, j]
                energy += np.sum(sudoku[i, :] == value) + np.sum(sudoku[:, j] == value) - 2
        return energy

    def mcmc(self, t=0.15, max_iter=50000):
        self.fill_sudoku()
        number_of_acceptance = 0

        a = 0.95
        add = 1

        energy = self.likelihood(self.grid)
        if energy == 0:
            return None

        sudoku_trial = copy.deepcopy(self.grid)
        best_energy = energy
        evolution_of_energy = [energy]

        nb_of_false = 0

        for i in range(max_iter):
            square = np.random.randint(0, 9)
            first_element = np.random.randint(0, 9)
            while self.grid_config[square // 3 * 3 + first_element // 3, first_element % 3 + (3 * square) % 9] == 1:
                square = np.random.randint(0, 9)
                first_element = np.random.randint(0, 9)
            second_element = np.random.randint(0, 9)
            while first_element == second_element or self.grid_config[
                square // 3 * 3 + second_element // 3, second_element % 3 + (3 * square) % 9
            ] == 1:
                second_element = np.random.randint(0, 9)

            sudoku_trial = copy.deepcopy(self.grid)

            tmp = sudoku_trial[square // 3 * 3 + first_element // 3, first_element % 3 + (3 * square) % 9]
            sudoku_trial[square // 3 * 3 + first_element // 3, first_element % 3 + (3 * square) % 9] = sudoku_trial[
                square // 3 * 3 + second_element // 3, second_element % 3 + (3 * square) % 9
            ]
            sudoku_trial[square // 3 * 3 + second_element // 3, second_element % 3 + (3 * square) % 9] = tmp

            e_trial = self.likelihood(sudoku_trial)
            if i % 100 == 0:
                evolution_of_energy.append(e_trial)

            u = np.random.uniform(0, 1)

            if e_trial < energy or u < np.exp(-(e_trial - energy) / t):
                self.grid = copy.deepcopy(sudoku_trial)
                energy = e_trial
                number_of_acceptance += 1
                t = t * a
                nb_of_false = 0
            else:
                nb_of_false += 1
                if nb_of_false >= 100:
                    add = 10
                    t += add
                    nb_of_false = 0

            if energy < best_energy:
                best_energy = energy

            if best_energy == 0:
                self.solved = 1
                break
        return None

    def is_on_row(self, sudoku, k, i):
        for j in range(9):
            if sudoku[i, j] == k:
                return True
        return False

    def is_on_column(self, sudoku, k, j):
        for i in range(9):
            if sudoku[i, j] == k:
                return True
        return False

    def is_on_block(self, sudoku, k, i, j):
        first_row = i - i % 3
        first_col = j - j % 3
        for row in range(first_row, first_row + 3):
            for col in range(first_col, first_col + 3):
                if sudoku[row, col] == k:
                    return True
        return False

    def is_valid(self, sudoku, k, i, j):
        if self.is_on_row(sudoku, k, i) is False and self.is_on_column(sudoku, k, j) is False and self.is_on_block(sudoku, k, i, j) is False:
            return True
        return False

    def solve_backtracking(self, sudoku, position=0):
        sudoku_trial = copy.deepcopy(sudoku)
        if position == 81:
            self.grid = copy.deepcopy(sudoku_trial)
            self.solved = 1
            return True
        i = position // 9
        j = position % 9
        if self.grid_config[i, j] == 1:
            return self.solve_backtracking(sudoku_trial, position + 1)
        else:
            for k in range(1, 10):
                if self.is_valid(sudoku_trial, k, i, j):
                    sudoku_trial[i, j] = k
                    if self.solve_backtracking(sudoku_trial, position + 1) is True:
                        return True
        return False

    def solve(self, method="backtracking"):
        start_time = time.time()
        if method == "backtracking":
            self.solve_backtracking(self.grid)
        elif method == "mcmc":
            self.mcmc()
        else:
            print("Unknown method")
        elapsed_time = time.time() - start_time
        if self.solved == 1:
            print("Solved in %0.2f seconds with the %s method" % (elapsed_time, method))
            print('Completed grid:\n')
            self.print_sudoku()
        else:
            print("No solution find in %0.2f seconds with %s method" % (elapsed_time, method))


if __name__ == '__main__':
    
    sd = Sudoku()
    sd.solve("backtracking")

    sd = Sudoku()
    sd.solve("mcmc")
