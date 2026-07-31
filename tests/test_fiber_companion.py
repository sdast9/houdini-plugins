"""PolyFEM output fiber companion and ReadPVD fallback integration."""

import os
import tempfile

import hou
import numpy as np

from test_polyfem_hda import BASE, make_cube_msh


def write_tiny_pvd(folder):
    points = "0 0 0  1 0 0  0 1 0  0 0 1  1 1 1"
    vtu = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="5" NumberOfCells="2">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
{points}</DataArray></Points>
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3  1 2 3 4</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">4 8</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">10 10</DataArray>
</Cells>
<PointData>
<DataArray type="Float64" Name="solution" NumberOfComponents="3"
format="ascii">{"0 0 0  " * 5}</DataArray>
<DataArray type="Float64" Name="body_ids"
format="ascii">1001 1001 1001 1001 1001</DataArray>
</PointData></Piece></UnstructuredGrid></VTKFile>"""
    with open(os.path.join(folder, "f0.vtu"), "w") as handle:
        handle.write(vtu)
    pvd = os.path.join(folder, "result.pvd")
    with open(pvd, "w") as handle:
        handle.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection><DataSet timestep="0" file="f0.vtu"/></Collection></VTKFile>""")
    return pvd


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_fiber_companion_")
    os.makedirs(os.path.join(work, "input"))
    mesh = os.path.join(work, "cube.msh")
    make_cube_msh(mesh)

    pre = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "fiber_companion_export")
    pre.setParms({"working_dir": work + "/", "file_location1": mesh})
    pre.parm("file_location1").pressButton()
    phm = pre.hdaModule()
    pre.setParms({
        "materials1_1": phm.MATERIAL_TOKENS.index("HGOFiber"),
        "fib_source1_1": "constant", "minimal_fields": 1,
        "materials_fields": 0})
    pre.parmTuple("fib_dir1_1").set((0, 1, 0))
    params_path = phm.build_params(pre)
    assert os.path.isfile(params_path)

    companion = os.path.join(
        work, "output", "polyfem_fiber_families.npz")
    assert os.path.isfile(companion), companion
    with np.load(companion, allow_pickle=False) as archive:
        assert archive["schema_version"].tolist() == [2]
        assert archive["direction_frame"].tolist() == [
            "simulation_world_reference_a0"]
        assert archive["directions"].shape[0] == 1
        values = archive["directions"][0]
        nonzero = np.linalg.norm(values, axis=1) > 1e-12
        assert np.allclose(values[nonzero], [0, 1, 0])
        assert archive["family_body_ids"].tolist() == [1001]
    print("PASS: preprocessing writes constant fiber families to output")

    # Verify a manually selected fiber file is staged and the parameter points
    # at the input copy (the callback is what the UI invokes).
    external = os.path.join(work, "external_fibers.txt")
    count = phm.volume_element_count(pre, 1)
    np.savetxt(external, np.tile([1.0, 0.0, 0.0], (count, 1)))
    pre.setParms({"fib_source1_1": "file", "fib_file1_1": external})
    phm.material_file_changed({
        "node": pre, "parm": pre.parm("fib_file1_1"),
        "script_multiparm_index2": "1",
        "script_multiparm_index": "1"})
    staged = pre.evalParm("fib_file1_1")
    assert os.path.dirname(staged) == os.path.join(work, "input")
    assert os.path.isfile(staged)
    print("PASS: imported fiber file is staged in input and read there")

    # Replace the fixture with a tiny result and matching companion to exercise
    # ReadPVD's fallback independently of PolyFEM material-field output.
    pvd = write_tiny_pvd(os.path.join(work, "output"))
    centroids = np.asarray([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])
    directions = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    with open(companion + ".tmp", "wb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray([1], dtype=np.int32),
            centroids=centroids, body_ids=np.asarray([1001, 1001]),
            family_names=np.asarray(
                ["hda_g1_v1_f0_fiber_direction"], dtype=np.str_),
            family_labels=np.asarray(
                ["Geometry 1 / Subdomain 1 / HGOFiber family 0"],
                dtype=np.str_),
            family_body_ids=np.asarray([1001], dtype=np.int64),
            directions=directions)
    os.replace(companion + ".tmp", companion)

    viewer = hou.node("/obj").createNode("readPVD::1.0", "fiber_fallback")
    viewer.setParms({"PVD_file": pvd, "source_block": "Volume"})
    viewer.hdaModule().start({"node": viewer})
    output = viewer.node("output")
    output.cook(force=True)
    geo = output.geometry()
    families = geo.attribValue("readpvd_fiber_families").split()
    assert families == ["hda_g1_v1_f0_fiber_direction"], families
    assert geo.attribValue("readpvd_fiber_reference_frame") == \
        "simulation_world_reference_a0"
    assert viewer.evalParm("has_fiber_data") == 1
    vectors = np.asarray(
        geo.pointFloatAttribValues(families[0])).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)
    assert {tuple(row) for row in np.unique(vectors, axis=0)} >= {
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}

    viewer.setParms({
        "show_fibers": 1, "fiber_frame": "reference",
        "fiber_stride": 1, "smooth_field": 0})
    output.cook(force=True)
    assert viewer.node("fiber_only").geometry().intrinsicValue(
        "primitivecount") == 5
    print("PASS: ReadPVD imports, exposes, and draws companion fiber families")

    # Production-sized no-scipy regression. The old fallback built large
    # target x source x 3 distance blocks and attempted billions of comparisons
    # for a 36k-element/144k-point result, which could terminate Houdini.
    count = 36067
    source = np.column_stack((
        np.arange(count, dtype=np.float64),
        np.arange(count, dtype=np.float64) % 137,
        np.arange(count, dtype=np.float64) % 29))
    target = source + 1e-6
    nearest = viewer.hdaModule()._nearest_rows(source, target)
    assert np.array_equal(nearest, np.arange(count))
    print("PASS: large companion projection avoids all-pairs allocation")
    print("workdir:", work)


if __name__ == "__main__":
    main()
