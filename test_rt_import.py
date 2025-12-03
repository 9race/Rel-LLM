#!/usr/bin/env python3
"""
Quick test to verify RT adapter can be imported and initialized.
"""

print("Testing RT adapter import...")

try:
    print("1. Importing rt_adapter...")
    from rt_adapter import RTEncoderOnly, HeteroDataToRTBatch, aggregate_cells_to_nodes
    print("   ✓ rt_adapter imported successfully")
    
    print("2. Checking relational-transformer path...")
    from pathlib import Path
    rt_path = Path(__file__).parent / "relational-transformer"
    if rt_path.exists():
        print(f"   ✓ Found relational-transformer at {rt_path}")
    else:
        print(f"   ✗ relational-transformer not found at {rt_path}")
        exit(1)
    
    print("3. Importing RelationalTransformer from rt.model...")
    import sys
    sys.path.insert(0, str(rt_path))
    from rt.model import RelationalTransformer
    print("   ✓ RelationalTransformer imported successfully")
    
    print("4. Testing RT model initialization...")
    rt_model = RelationalTransformer(
        num_blocks=2,  # Small for testing
        d_model=128,   # Small for testing
        d_text=384,
        num_heads=4,   # Small for testing
        d_ff=512,      # Small for testing
    )
    print("   ✓ RT model initialized successfully")
    
    print("5. Testing RTEncoderOnly wrapper...")
    rt_encoder = RTEncoderOnly(rt_model)
    print("   ✓ RTEncoderOnly wrapper created successfully")
    
    print("\n✅ All RT imports and initialization tests passed!")
    print("RT integration is ready to use.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)




