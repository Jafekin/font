"""
Test script to verify the ancient book processor implementation.
Run this to ensure all modules work correctly.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from pre_process import AncientBookProcessor, StructureDetector, DistortionHandler, LineInfoExtractor
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_processor_initialization():
    """Test processor initialization."""
    print("\nTesting processor initialization...")

    try:
        from pre_process import AncientBookProcessor

        # Initialize with CPU
        processor = AncientBookProcessor(device='cpu')
        print(f"✓ Processor initialized on CPU")

        # Get device info
        info = processor.get_device_info()
        print(f"  Device: {info['device']}")
        print(f"  PyTorch Version: {info['pytorch_version']}")
        print(f"  OpenCV Version: {info['opencv_version']}")

        return True

    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False


def test_structure_detector():
    """Test structure detector."""
    print("\nTesting structure detector...")

    try:
        import numpy as np
        from pre_process import StructureDetector

        # Create a dummy image
        image = np.random.randint(0, 256, (500, 400, 3), dtype=np.uint8)

        detector = StructureDetector(device='cpu')
        result = detector.detect_structures(image)

        # Check result structure
        assert 'version_heart' in result
        assert 'column_lines' in result
        assert 'page_numbers' in result
        assert 'title_labels' in result
        assert 'borders' in result
        assert 'text_regions' in result

        print("✓ Structure detector working correctly")
        print(f"  - Version heart found: {result['version_heart']['found']}")
        print(f"  - Column lines detected: {len(result['column_lines'])}")
        print(f"  - Page numbers found: {len(result['page_numbers'])}")
        print(f"  - Title labels found: {len(result['title_labels'])}")
        print(f"  - Text regions: {len(result['text_regions'])}")

        return True

    except Exception as e:
        print(f"✗ Structure detector error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distortion_handler():
    """Test distortion handler."""
    print("\nTesting distortion handler...")

    try:
        import numpy as np
        from pre_process import DistortionHandler

        # Create a dummy RGB image with some patterns
        image = np.random.randint(100, 200, (500, 400, 3), dtype=np.uint8)
        # Add some darker areas (stains)
        image[100:120, 100:120] = 50

        handler = DistortionHandler(device='cpu')
        result = handler.process_distortions(image)

        # Check result structure
        assert 'cleaned_image' in result
        assert 'stain_mask' in result
        assert 'blur_map' in result
        assert 'quality_score' in result

        print("✓ Distortion handler working correctly")
        print(f"  - Quality score: {result['quality_score']:.2%}")
        print(f"  - Stains detected: {len(result['stains_info'])}")
        print(f"  - Blur regions: {len(result['blur_regions'])}")
        print(f"  - Fold marks: {len(result['fold_marks'])}")

        return True

    except Exception as e:
        print(f"✗ Distortion handler error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_line_extractor():
    """Test line info extractor."""
    print("\nTesting line info extractor...")

    try:
        import numpy as np
        from pre_process import LineInfoExtractor

        # Create a dummy image
        image = np.random.randint(100, 200, (500, 400), dtype=np.uint8)

        extractor = LineInfoExtractor(device='cpu')
        result = extractor.extract_line_info(image)

        # Check result structure
        assert 'num_lines' in result
        assert 'chars_per_line' in result
        assert 'line_positions' in result
        assert 'borders' in result
        assert 'layout_type' in result

        print("✓ Line extractor working correctly")
        print(f"  - Lines detected: {result['num_lines']}")
        print(f"  - Chars per line: {result['chars_per_line']}")
        print(f"  - Line spacing: {result['line_spacing']:.1f}")
        print(f"  - Layout type: {result['layout_type']}")
        print(f"  - Confidence: {result['confidence']:.2%}")

        return True

    except Exception as e:
        print(f"✗ Line extractor error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full processing pipeline."""
    print("\nTesting full processing pipeline...")

    try:
        import numpy as np
        from pre_process import AncientBookProcessor

        # Create a dummy ancient book-like image
        # White background with some text-like patterns
        image = np.ones((600, 500, 3), dtype=np.uint8) * 240  # Off-white background

        # Add some text-like regions (dark areas)
        for i in range(10):
            y_pos = 50 + i * 50
            image[y_pos:y_pos+30, 50:450] = 50  # Dark text areas

        # Add some noise/stains
        image[100:110, 100:110] = 30
        image[300:310, 300:310] = 25

        # Process
        processor = AncientBookProcessor(device='cpu')
        result = processor.process(image, full_analysis=True)

        # Check result
        assert result['success'], f"Processing failed: {result.get('error')}"

        # Check all expected fields
        assert 'structures' in result
        assert 'distortions' in result
        assert 'line_info' in result
        assert 'summary' in result

        print("✓ Full pipeline working correctly")

        # Print summary
        summary = result['summary']
        print(f"\n  Summary:")
        print(f"    Page type: {summary['page_type']}")
        print(f"    Layout: {summary['layout_info']['layout_type']}")
        print(f"    Lines: {summary['layout_info']['num_lines']}")
        print(f"    Chars/line: {summary['layout_info']['chars_per_line']}")
        print(f"    Quality: {summary['image_quality']['overall_score']:.1%}")
        print(f"    Recommendations: {len(summary['recommendations'])}")

        return True

    except Exception as e:
        print(f"✗ Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Ancient Book Processor - Test Suite")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Processor Initialization", test_processor_initialization),
        ("Structure Detector", test_structure_detector),
        ("Distortion Handler", test_distortion_handler),
        ("Line Info Extractor", test_line_extractor),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

