from scripts.report_valve_panel import structure_breadth


def test_unexecuted_plan_cannot_fill_module_breadth_or_band_success():
    rule=dict(module_bands=[[1,2],[3,5],[6,8]],minimum_successes_in_each_module_band=1,
              minimum_distinct_executed_module_counts=3)
    rows=[dict(modules=1,executed=True,success=True),dict(modules=4,executed=True,success=True),
          dict(modules=8,executed=False,success=False)]
    report=structure_breadth(rows,rule)
    assert report['distinct_executed']==report['distinct_successful']==2
    assert not report['passes']
    rows[-1].update(executed=True)
    report=structure_breadth(rows,rule)
    assert report['distinct_executed']==3 and report['distinct_successful']==2
    assert not report['passes']
    rows[-1]['success']=True
    assert structure_breadth(rows,rule)['passes']
