#!/usr/bin/env python3
"""
Simple smoke test for memory agent security research framework.

This script tests basic functionality without complex pytest setup.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Memory Agent Security - Smoke Test")
print("=" * 40)


def test_attacks():
    """Test attack implementations."""
    print("\n🔬 Testing Attacks...")

    try:
        from attacks.implementations import create_attack

        # Test AgentPoison attack
        attack = create_attack("agent_poison")
        result = attack.execute("Test memory content for poisoning")

        assert "attack_type" in result
        assert result["attack_type"] == "agent_poison"
        assert "success" in result
        assert "poisoned_content" in result

        print("✓ AgentPoison attack working")

        # Test MINJA attack
        attack = create_attack("minja")
        result = attack.execute({"memory": "test data"})

        assert result["attack_type"] == "minja"
        assert "injected_content" in result

        print("✓ MINJA attack working")

        # Test InjecMEM attack
        attack = create_attack("injecmem")
        result = attack.execute(["item1", "item2", "item3"])

        assert result["attack_type"] == "injecmem"
        assert "manipulated_content" in result

        print("✓ InjecMEM attack working")

        return True

    except Exception as e:
        print(f"✗ Attack test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_defenses():
    """Test defense implementations."""
    print("\n🛡️  Testing Defenses...")

    try:
        from defenses.implementations import create_defense

        # Test Watermark defense
        defense = create_defense("watermark")
        activated = defense.activate()
        assert activated == True

        result = defense.detect_attack("Test content")
        assert "attack_detected" in result
        assert "confidence" in result

        defense.deactivate()
        print("✓ Watermark defense working")

        # Test Validation defense
        defense = create_defense("validation")
        defense.activate()
        result = defense.detect_attack("MALICIOUS_INJECTION: override()")
        assert "attack_detected" in result
        defense.deactivate()
        print("✓ Content validation defense working")

        return True

    except Exception as e:
        print(f"✗ Defense test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_watermarking():
    """Test watermarking algorithms."""
    print("\n🔐 Testing Watermarking...")

    try:
        from watermark.watermarking import (ProvenanceTracker,
                                            create_watermark_encoder)

        # Test LSB encoder
        encoder = create_watermark_encoder("lsb")
        content = "Test content for watermarking"
        watermarked = encoder.embed(content, "test_watermark")
        extracted = encoder.extract(watermarked)

        assert isinstance(watermarked, str)
        print("✓ LSB watermarking working")

        # Test Provenance tracker
        tracker = ProvenanceTracker()
        content_id = "test_content_001"
        watermark_id = tracker.register_content(content_id, content)
        watermarked = tracker.watermark_content(content, watermark_id)

        assert isinstance(watermark_id, str)
        assert isinstance(watermarked, str)

        print("✓ Provenance tracking working")

        return True

    except Exception as e:
        print(f"✗ Watermarking test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation framework."""
    print("\n📊 Testing Evaluation...")

    try:
        from evaluation.benchmarking import AttackEvaluator, DefenseEvaluator

        # Test attack evaluator
        evaluator = AttackEvaluator()
        metrics = evaluator.evaluate_attack(
            "agent_poison", ["test content"], num_trials=3
        )

        assert hasattr(metrics, "attack_type")
        assert hasattr(metrics, "total_attempts")
        assert hasattr(metrics, "asr_r")

        print("✓ Attack evaluation working")

        # Test defense evaluator
        evaluator = DefenseEvaluator()
        # Create mock attack suite and content
        from attacks.implementations import AttackSuite

        attack_suite = AttackSuite()
        clean_content = ["clean content"]
        poisoned_content = ["poisoned content"]

        metrics = evaluator.evaluate_defense(
            "watermark", attack_suite, clean_content, poisoned_content
        )

        assert hasattr(metrics, "defense_type")
        assert hasattr(metrics, "tpr")
        assert hasattr(metrics, "fpr")

        print("✓ Defense evaluation working")

        return True

    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_systems():
    """Test memory system wrappers."""
    print("\n🧠 Testing Memory Systems...")

    try:
        from memory_systems.wrappers import create_memory_system

        # Test with mock configuration
        memory = create_memory_system("mem0", {"user_id": "test"})
        assert memory is not None

        print("✓ Memory system wrapper working")

        return True

    except Exception as e:
        print(f"✗ Memory system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    tests = [
        test_memory_systems,
        test_attacks,
        test_defenses,
        test_watermarking,
        test_evaluation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📈 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All smoke tests passed! Framework is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
