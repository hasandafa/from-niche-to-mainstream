#!/usr/bin/env python3
"""
Ultimate Jupyter Notebook Recovery Tool
Handles severely corrupted notebooks with truncated JSON, invalid control chars, etc.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def clean_content(content: str) -> str:
    """Remove invalid control characters."""
    return ''.join(c for c in content if ord(c) >= 32 or c in '\n\r\t')

def extract_cells_from_corrupted(content: str) -> List[Dict[str, Any]]:
    """Extract individual cells even from heavily corrupted notebooks."""
    cells = []
    
    # Pattern to find cell boundaries
    # Cells typically start with {"cell_type": "code" or "markdown"
    cell_pattern = r'\{\s*"cell_type"\s*:\s*"(code|markdown)"'
    
    matches = list(re.finditer(cell_pattern, content, re.IGNORECASE))
    print(f"   Found {len(matches)} potential cells")
    
    for i, match in enumerate(matches):
        start = match.start()
        
        # Try to find the end of this cell
        # Look for the next cell start or end of array
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            # Last cell - look for closing bracket
            end = content.find(']', start)
            if end == -1:
                end = len(content)
        
        # Extract cell JSON
        cell_json = content[start:end].rstrip(',').strip()
        
        # Try to fix common issues
        if not cell_json.endswith('}'):
            # Try to close the cell properly
            cell_json = cell_json.rstrip() + '}'
        
        # Try to parse
        try:
            cell = json.loads(cell_json)
            cells.append(cell)
        except json.JSONDecodeError:
            # Cell is too corrupted, create a basic one
            cell_type = match.group(1)
            
            # Try to extract at least the source
            source_match = re.search(r'"source"\s*:\s*(\[.*?\]|\".*?\")', cell_json, re.DOTALL)
            source = []
            
            if source_match:
                try:
                    source_str = source_match.group(1)
                    source = json.loads(source_str)
                    if isinstance(source, str):
                        source = [source]
                except:
                    source = ["# Content could not be recovered"]
            
            cell = create_cell(cell_type, source)
            cells.append(cell)
            print(f"   ⚠ Cell {i+1}: Reconstructed from fragments")
    
    return cells

def create_cell(cell_type: str, source: List[str] = None) -> Dict[str, Any]:
    """Create a valid cell structure."""
    if source is None:
        source = []
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    return cell

def create_notebook_template() -> Dict[str, Any]:
    """Create a minimal valid notebook."""
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def recover_notebook(input_path: str, output_path: str = None) -> bool:
    """
    Attempt to recover a corrupted Jupyter notebook.
    
    Args:
        input_path: Path to corrupted notebook
        output_path: Where to save recovered notebook (default: adds _recovered suffix)
    
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_recovered.ipynb"
    else:
        output_path = Path(output_path)
    
    print(f"\n🔧 Attempting to recover: {input_path.name}")
    print(f"=" * 60)
    
    # Read file
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return False
    
    print(f"📄 File size: {len(content):,} bytes")
    
    # Clean content
    print("🧹 Cleaning control characters...")
    content = clean_content(content)
    
    # Try standard JSON parse first
    print("🔍 Attempting standard JSON parse...")
    try:
        notebook = json.loads(content)
        if 'cells' in notebook and isinstance(notebook['cells'], list):
            print("✅ File is actually valid! Saving cleaned version...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            print(f"💾 Saved to: {output_path}")
            return True
    except json.JSONDecodeError as e:
        print(f"⚠ JSON parse failed: {str(e)[:100]}")
        print("   Attempting cell-by-cell recovery...")
    
    # Extract cells from corrupted content
    cells = extract_cells_from_corrupted(content)
    
    if not cells:
        print("\n❌ Could not recover any cells from the file")
        print("\n💡 The file may be too corrupted. Consider:")
        print("   1. Using a backup if available")
        print("   2. Recreating the notebook from scratch")
        print("   3. Checking if you have the .ipynb_checkpoints version")
        return False
    
    # Create new notebook with recovered cells
    print(f"\n✅ Recovered {len(cells)} cells")
    notebook = create_notebook_template()
    notebook['cells'] = cells
    
    # Save
    print(f"\n💾 Saving recovered notebook...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✅ Successfully saved to: {output_path}")
        
        # Show stats
        code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
        md_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
        
        print(f"\n📊 Recovery Summary:")
        print(f"   Total cells: {len(cells)}")
        print(f"   Code cells: {code_cells}")
        print(f"   Markdown cells: {md_cells}")
        print(f"\n✨ You can now open: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def batch_recover(directory: str, pattern: str = "*.ipynb"):
    """Recover all notebooks in a directory."""
    directory = Path(directory)
    notebooks = list(directory.glob(pattern))
    
    print(f"\n🔍 Found {len(notebooks)} notebooks in {directory}")
    print("=" * 60)
    
    results = []
    for nb_path in notebooks:
        if '_recovered' in nb_path.name or '_fixed' in nb_path.name:
            print(f"⏭ Skipping already recovered: {nb_path.name}")
            continue
        
        success = recover_notebook(nb_path)
        results.append((nb_path.name, success))
        print()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BATCH RECOVERY SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, s in results if s)
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {len(results) - successful}/{len(results)}")
    
    if successful < len(results):
        print("\n❌ Failed files:")
        for name, success in results:
            if not success:
                print(f"   - {name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Jupyter Notebook Recovery Tool")
        print("=" * 60)
        print("\nUsage:")
        print("  Single file:  python fix_notebook.py <notebook.ipynb>")
        print("  Directory:    python fix_notebook.py --batch <directory>")
        print("\nExamples:")
        print("  python fix_notebook.py 01_data_exploration.ipynb")
        print("  python fix_notebook.py --batch notebooks/")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("❌ Please specify a directory for batch recovery")
            sys.exit(1)
        batch_recover(sys.argv[2])
    else:
        success = recover_notebook(sys.argv[1])
        sys.exit(0 if success else 1)