import models.solver

if __name__ == '__main__':
    solver = models.solver.KaggleCompSolver(
        comp_name='comp2',
        train_label_pos=-1,
        comp_method_type='supervised-classification',
        comp_task_type='data',
        comp_algo='knn'
    )

    solver.info()
    solver.data_parse(test_columns_drop=['S.No'])
    solver.info(logs=True)

    solver.solve(metric_type='f1')
