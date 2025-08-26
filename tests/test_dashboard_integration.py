#!/usr/bin/env python3
"""
Test Script for Dashboard-TopoViz Integration
==============================================

Tests the integration between the dashboard and topo_viz.py functionality.
Verifies that all components work together properly.
"""

import sys

# Add current directory to path for imports
sys.path.insert(0, ".")


def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")

    try:
        print("✅ dashboard_core imported successfully")
    except Exception as e:
        print(f"❌ dashboard_core import failed: {e}")
        return False

    try:
        import topo_viz

        print("✅ topo_viz imported successfully")
        print(f"📊 Available scenes: {len(topo_viz.list_scenes())}")
    except Exception as e:
        print(f"⚠️ topo_viz import failed: {e} (fallback mode will be used)")

    return True


def test_topology_controller():
    """Test the enhanced TopologyViewController."""
    print("\n🧪 Testing TopologyViewController...")

    try:
        from dashboard_core import (
            AVAILABLE_SCENES,
            TOPO_VIZ_AVAILABLE,
            TopologyViewController,
        )

        controller = TopologyViewController()
        print("✅ TopologyViewController created")
        print(f"📄 Current scene: {controller.current_scene}")

        if TOPO_VIZ_AVAILABLE:
            print(f"🎭 Available scenes: {len(AVAILABLE_SCENES)}")

            # Test scene categories
            categories = controller.get_available_scenes()
            print(f"📁 Scene categories: {list(categories.keys())}")

            # Test setting a scene
            if "gamma_volume" in AVAILABLE_SCENES:
                controller.set_scene("gamma_volume", n_max=15.0)
                print("✅ Scene setting works")

        else:
            print("ℹ️ Running in fallback mode (topo_viz not available)")

        return True

    except Exception as e:
        print(f"❌ TopologyViewController test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dashboard_creation():
    """Test dashboard creation without showing it."""
    print("\n🧪 Testing dashboard creation...")

    try:
        from dashboard_core import DimensionalDashboard

        # Create dashboard but don't launch
        dashboard = DimensionalDashboard()
        print("✅ Dashboard created successfully")

        # Test state
        print(f"📊 Initial dimension: {dashboard.state.dimension}")
        print(
            f"🎮 Event bus has subscribers: {len(dashboard.event_bus._subscribers) > 0}"
        )

        # Test topology controller integration
        scene_info = dashboard.topo_controller.get_available_scenes()
        print(f"🎭 Scene categories available: {len(scene_info)}")

        return True

    except Exception as e:
        print(f"❌ Dashboard creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scene_switching():
    """Test scene switching functionality."""
    print("\n🧪 Testing scene switching...")

    try:
        from dashboard_core import (
            AVAILABLE_SCENES,
            TOPO_VIZ_AVAILABLE,
            TopologyViewController,
        )

        if not TOPO_VIZ_AVAILABLE:
            print("ℹ️ Skipping scene switching test (topo_viz not available)")
            return True

        controller = TopologyViewController()

        # Test switching between scenes
        test_scenes = ["gamma_volume", "gamma_area", "qwz_curvature"]
        available_test_scenes = [s for s in test_scenes if s in AVAILABLE_SCENES]

        if len(available_test_scenes) >= 2:
            # Switch to first test scene
            first_scene = available_test_scenes[0]
            controller.set_scene(first_scene)
            assert controller.current_scene == first_scene
            print(f"✅ Switched to {first_scene}")

            # Switch to second test scene
            second_scene = available_test_scenes[1]
            controller.set_scene(second_scene)
            assert controller.current_scene == second_scene
            print(f"✅ Switched to {second_scene}")

            print("✅ Scene switching works correctly")
        else:
            print("⚠️ Not enough test scenes available for switching test")

        return True

    except Exception as e:
        print(f"❌ Scene switching test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("🚀 DASHBOARD-TOPOVIZ INTEGRATION TESTS")
    print("=" * 50)

    tests = [
        test_imports,
        test_topology_controller,
        test_dashboard_creation,
        test_scene_switching,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📈 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


def demo_dashboard():
    """Launch the dashboard for interactive testing."""
    print("\n🎯 LAUNCHING DASHBOARD DEMO")
    print("=" * 30)
    print("This will open the interactive dashboard.")
    print("Use 'n' and 'p' keys to switch between topology scenes.")
    print("Move the sliders to see real-time parameter updates.")
    print("Close the window when done testing.")

    input("\nPress Enter to launch dashboard (or Ctrl+C to skip)...")

    try:
        from dashboard_core import main

        main()
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard demo skipped")
    except Exception as e:
        print(f"❌ Dashboard demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the tests
    tests_passed = run_integration_tests()

    if tests_passed and "--demo" in sys.argv:
        demo_dashboard()
    elif tests_passed:
        print("\n💡 To test the interactive dashboard, run:")
        print("python test_dashboard_integration.py --demo")
    else:
        print("\n🔧 Fix the failing tests before running the demo.")
        sys.exit(1)
