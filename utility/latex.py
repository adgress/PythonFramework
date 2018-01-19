import numpy as np


def cell_to_latex_str(*args):
    assert 0 < len(args) <= 2
    num_str = '%.2f' % args[0]
    s = '$' + num_str
    if len(args) == 2:
        num_str = '%.2f' % args[1]
        s += ' (' + num_str + ')'
    s += '$'
    return s

def ndarray_to_table(values, first_row=None, cell_format_func=cell_to_latex_str, precision=4):
    latex_str = ''
    for row_idx, rows in enumerate(zip(*values)):
        if first_row is not None:
            latex_str += first_row[row_idx]
        is_first = True

        if len(rows) == 1:
            for value in rows[0]:
                if not (is_first and first_row is None):
                    latex_str += ' & '
                is_first = False
                latex_str += cell_format_func(value)
        else:
            for values in zip(*rows):
                if not (is_first and first_row is None):
                    latex_str += ' & '
                is_first = False
                latex_str += cell_format_func(*values)

        latex_str += str(' \\\\ \n')
        latex_str += str('\\hline \n')
    return latex_str


if __name__ == '__main__':
    values1 = np.eye(10)
    values2 = np.ones((10, 10))
    first_row_strs = ['Name']*10
    print ndarray_to_table((values1, ), first_row_strs)