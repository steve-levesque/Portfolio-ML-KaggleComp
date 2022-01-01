import models.solver

if __name__ == '__main__':
    solver = models.solver.KaggleCompSolver(
        comp_name='comp1',
        train_label_pos=-1,
        comp_method_type='supervised-classification',
        comp_task_type='data',
        comp_algo='xgb'
    )

    solver.info()
    solver.data_parse(test_columns_drop=['S.No'])
    solver.info(logs=True)

    solver.solve(objective='multi:softmax', metric_type='acc')
