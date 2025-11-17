"""
Example script demonstrating the ancient book processor usage.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pre_process import AncientBookProcessor


def process_single_image(image_path: str, output_dir: str = None):
    """
    Process a single ancient book image.

    Args:
        image_path: Path to the image file
        output_dir: Directory to save results (optional)
    """
    print(f"Processing image: {image_path}")

    # Initialize processor
    processor = AncientBookProcessor(use_cuda=True)

    # Print device info
    device_info = processor.get_device_info()
    print(f"\nDevice Info:")
    print(f"  Device: {device_info['device']}")
    print(f"  PyTorch Version: {device_info['pytorch_version']}")
    print(f"  OpenCV Version: {device_info['opencv_version']}")
    if device_info['cuda_available']:
        print(f"  CUDA Devices: {device_info['device_count']}")

    # Process image
    print(f"\nProcessing...")
    result = processor.process(image_path, full_analysis=True)

    # Print results
    if result['success']:
        print("\n✓ Processing completed successfully!")

        # Print summary
        summary = result['summary']
        print(f"\nAnalysis Summary:")
        print(f"  Page Type: {summary['page_type']}")
        print(f"  Layout Type: {summary['layout_info']['layout_type']}")
        print(f"  Number of Lines: {summary['layout_info']['num_lines']}")
        print(f"  Characters per Line: {summary['layout_info']['chars_per_line']}")
        print(f"  Image Quality Score: {summary['image_quality']['overall_score']:.2%}")
        print(f"  Stains Detected: {summary['image_quality']['stain_count']}")
        print(f"  Blur Regions: {summary['image_quality']['blur_regions']}")
        print(f"  Fold Marks: {summary['image_quality']['fold_marks']}")

        print(f"\nStructural Elements:")
        structs = summary['structural_elements']
        print(f"  Has Version Heart: {structs['has_version_heart']}")
        print(f"  Column Lines: {structs['column_line_count']}")
        print(f"  Page Numbers Found: {structs['page_number_locations']}")
        print(f"  Title Labels: {structs['title_labels']}")
        print(f"  Border Type: {structs['border_type']}")

        print(f"\nRecommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")

        # Save detailed results if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{Path(image_path).stem}_analysis.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to serializable format
            serializable_result = _make_serializable(result)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            print(f"\nDetailed results saved to: {output_path}")
    else:
        print(f"\n✗ Processing failed: {result['error']}")


def process_directory(image_dir: str, output_dir: str = None):
    """
    Process all images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save results (optional)
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        return

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} image files")

    # Process each image
    results_summary = []
    processor = AncientBookProcessor(use_cuda=True)

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")

        try:
            result = processor.process(str(image_path), full_analysis=False)

            if result['success']:
                print(f"  ✓ Success")
                summary = result['summary']
                results_summary.append({
                    'filename': image_path.name,
                    'status': 'success',
                    'lines': summary['layout_info']['num_lines'],
                    'quality': f"{summary['image_quality']['overall_score']:.1%}"
                })
            else:
                print(f"  ✗ Failed: {result['error']}")
                results_summary.append({
                    'filename': image_path.name,
                    'status': 'failed',
                    'error': result['error']
                })

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results_summary.append({
                'filename': image_path.name,
                'status': 'error',
                'error': str(e)
            })

    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing Summary:")
    print(f"{'='*50}")

    success_count = sum(1 for r in results_summary if r['status'] == 'success')
    print(f"Successfully processed: {success_count}/{len(image_files)}")

    # Save summary report
    if output_dir:
        output_path = Path(output_dir) / "processing_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)

        print(f"Summary saved to: {output_path}")


def _make_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-compatible format."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process ancient book images using PyTorch-based preprocessing'
    )

    parser.add_argument(
        'image_path',
        help='Path to image file or directory'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output directory for results',
        default=None
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    if image_path.is_file():
        # Process single image
        process_single_image(str(image_path), args.output)

    elif image_path.is_dir():
        # Process directory
        process_directory(str(image_path), args.output)

    else:
        print(f"Error: Path not found: {image_path}")


if __name__ == '__main__':
    main()

