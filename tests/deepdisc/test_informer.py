import pytest
from rail.estimation.algos.deepdisc import DeepDiscInformer

def test_basic_informer_stage_creation():
    """A simple test to make sure that the Informer stage can be created"""

    deep_dict = dict(
        cfgfile = './foo.txt',
        batch_size=4,
        num_gpus=2,
    )

    informer_stage = DeepDiscInformer.make_stage(
        name="foo_DeepDISC",
        model="output_informer_model.pkl",
        **deep_dict
    )

    assert informer_stage.name == "DeepDiscInformer"


def test_batch_and_num_gpus_mismatch():
    """Test that an error is raised if batch_size is not an even multiple of num_gpus"""

    deep_dict = dict(
        cfgfile = './foo.txt',
        batch_size=5,
        num_gpus=2,
    )

    with pytest.raises(ValueError) as exc:
        _ = DeepDiscInformer.make_stage(
            name="Inform_DeepDISC",
            model="test_informer.pkl",
            **deep_dict
        )

    assert "must be an even multiple of num_gpus" in str(exc.value)
