import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from packaging.requirements import Requirement

spec=importlib.util.spec_from_file_location('wheel_plan',Path(__file__).resolve().parents[2]/'scripts/prepare_wall_rlt_wheels.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_extras_transitive_closure_uses_base_without_replacing_torch():
    distributions={'app':SimpleNamespace(version='1.0',requires=['torch>=2','core==3','extra==4; extra=="full"']),
                   'core':SimpleNamespace(version='3',requires=['app']),
                   'extra':SimpleNamespace(version='4',requires=[])}
    pins,base=module.overlay_closure([Requirement('app[full]==1.0')],{'torch':'2.10.0+cu128'},distributions.__getitem__)
    assert pins=={'app':'1.0','core':'3','extra':'4'}
    assert base=={'torch':'2.10.0+cu128'}


def test_conflicting_base_version_is_not_silently_replaced():
    with pytest.raises(ValueError,match='base conflict'):
        module.overlay_closure([Requirement('torch>=3')],{'torch':'2.10.0+cu128'})


def test_missing_extra_review_and_unverified_local_pin_fail():
    with pytest.raises(ValueError,match='extras need explicit review'):
        module.overlay_closure([Requirement('foo[bar]')],{'foo':'1'})
    with pytest.raises(ValueError,match='does not satisfy'):
        module.overlay_closure([Requirement('foo==2')],{},lambda n:SimpleNamespace(version='1'))
