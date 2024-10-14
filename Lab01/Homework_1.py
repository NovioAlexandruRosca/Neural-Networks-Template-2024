import copy
import math


def read_input(input_path):
    with open(input_path, "r") as file:
        content = file.read()
        return content.split("\n")


def clean_coefficient(value):
    cleaned_value = value[:-1].lstrip("+")

    if cleaned_value in ["", "-"]:
        return cleaned_value + "1" if cleaned_value == "" else "-1"

    return cleaned_value


def process_input(equations):
    free_terms = []
    coefficients = []
    if len(equations) < 3:
        raise Exception("Not enough equations")
    for equation in equations:
        leq, req = [side.strip() for side in equation.split("=")]

        leq_parts = leq.split()

        x_value = leq_parts[0]

        if len(leq_parts[1::]) % 2 == 1:
            raise ValueError(f"The left part of the equation is incorrectly formatted")

        leq_parts = leq_parts[1::]
        y_value, z_value = [
            leq_parts[i] + leq_parts[i + 1] for i in range(0, len(leq_parts) - 1, 2)
        ]

        clear_x_value = clean_coefficient(x_value)
        clear_y_value = clean_coefficient(y_value)
        clear_z_value = clean_coefficient(z_value)

        try:
            coefficients.append(
                [float(clear_x_value), float(clear_y_value), float(clear_z_value)]
            )
            free_terms.append(float(req))
        except ValueError:
            raise ValueError(f"{req}: is not a valid number.")

    return coefficients, free_terms


def calculate_determinant(matrix):
    if len(matrix) == 2:
        a11, a12 = matrix[0]
        a21, a22 = matrix[1]
        return a11 * a22 - a12 * a21

    elif len(matrix) == 3:
        a11, a12, a13 = matrix[0]
        a21, a22, a23 = matrix[1]
        a31, a32, a33 = matrix[2]

        return (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )


def calculate_trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def calculate_vector_norm(free_terms):
    return math.sqrt(
        sum(int(free_terms[i]) * int(free_terms[i]) for i in range(len(free_terms)))
    )


def calculate_transpose(matrix):
    transpose = []
    for j in range(len(matrix[0])):
        new_row = []
        for i in range(len(matrix)):
            new_row.append(matrix[i][j])

        transpose.append(new_row)

    return transpose


def calculate_mv_multiplication(matrix, free_terms):
    result = []
    for row in matrix:
        result.append(sum(row[i] * free_terms[i] for i in range(len(matrix))))

    return result


def solve_system(coefficients, free_terms, method=1):
    results = []
    base_matrix_determinant = calculate_determinant(coefficients)

    if base_matrix_determinant == 0:
        raise Exception("The determinant has to be >= 0")

    if method == 1:

        for i in range(3):
            new_matrix = calculate_transpose(copy.deepcopy(coefficients))
            new_matrix[i] = free_terms
            new_matrix_determinant = calculate_determinant(new_matrix)
            results.append(new_matrix_determinant / base_matrix_determinant)

    elif method == 2:

        cofactor_matrix = []

        for i in range(3):
            row = []
            for j in range(3):

                base_matrix = copy.deepcopy(coefficients)
                del base_matrix[i]
                transpose = calculate_transpose(base_matrix)
                del transpose[j]

                m_ij_determinant = calculate_determinant(transpose)
                row_element = math.pow(-1, i + j) * m_ij_determinant
                row.append(row_element)

            cofactor_matrix.append(row)

        cofactor_matrix = calculate_transpose(cofactor_matrix)
        results = calculate_mv_multiplication(cofactor_matrix, free_terms)
        results = [result * (1 / base_matrix_determinant) for result in results]

    return results


def starter(input_path):
    equations = read_input(input_path)
    coefficients, free_terms = process_input(equations)

    print(f"\nBase Matrix:")
    for i, row in enumerate(coefficients):
        print(row, "=", free_terms[i])

    transpose = calculate_transpose(coefficients)

    print(f"\nTranspose Matrix:")
    for i, row in enumerate(transpose):
        print(row, "=", free_terms[i])

    print(f"\nDeterminant: {calculate_determinant(coefficients)}")
    print(f"\nTrace: {calculate_trace(coefficients)}")
    print(f"\nVector Norm: {calculate_vector_norm(free_terms)}")
    print(
        f"\nM&V Multiplication: {calculate_mv_multiplication(coefficients, free_terms)}"
    )

    x, y, z = solve_system(coefficients, free_terms)
    print(f"\nResults with Cramer's method: X:{x}, Y:{y}, Z:{z}")

    x, y, z = solve_system(coefficients, free_terms, method=2)
    print(f"Results with Inversion method: X:{x}, Y:{y}, Z:{z}")


starter("./system.txt")
