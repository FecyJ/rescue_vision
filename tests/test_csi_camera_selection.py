from __future__ import annotations

import pytest

from rescue_vision.camera.csi import resolve_csi_camera_index


def test_physical_selection_is_independent_of_sensor_and_enumeration_order() -> None:
    cameras = [
        {'Num': 0, 'Model': 'imx219', 'Id': '/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10'},
        {'Num': 1, 'Model': 'imx708_wide', 'Id': '/base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a'},
    ]
    assert resolve_csi_camera_index(0, camera_info=cameras) == 1
    assert resolve_csi_camera_index(1, camera_info=cameras) == 0
    assert resolve_csi_camera_index(0, camera_info=list(reversed(cameras))) == 1


def test_only_camera_on_cam1_can_be_index_zero() -> None:
    assert resolve_csi_camera_index(1, camera_info=[
        {'Num': 0, 'Id': '/base/axi/pcie@1000120000/rp1/i2c@80000/imx708@1a'}
    ]) == 0


def test_missing_physical_port_reports_discovered_devices() -> None:
    cameras = [{'Num': 0, 'Id': '/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10'}]
    with pytest.raises(RuntimeError, match='CAM/DISP0.*imx219'):
        resolve_csi_camera_index(0, camera_info=cameras)


@pytest.mark.parametrize('port', [-1, 2, True])
def test_invalid_physical_port_is_rejected(port) -> None:
    with pytest.raises(ValueError, match='csi_port'):
        resolve_csi_camera_index(port, camera_info=[])
