#!/usr/bin/env python
"""
Quick Start: Handle Your Large Dataset

This script demonstrates the simplest ways to pre-process
and optimize your cancer classification project for large datasets.
"""

import os
import sys

# Add to Python path
sys.path.insert(0, os.path.dirname(__file__))

def quick_start_option_1():
    """Option 1: Pre-filter your CSV using built-in utilities."""
    print("\n" + "="*60)
    print("OPTION 1: Pre-filter CSV (Easiest)")
    print("="*60)
    
    from large_dataset_utils import LargeDatasetHandler
    
    handler = LargeDatasetHandler()
    
    # Just change these values:
    input_file = "your_data.csv"  # ← Your large CSV file
    output_file = "your_data_filtered.csv"  # ← Will be created
    
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    print("\nProcessing...")
    
    df = handler.load_and_filter_csv(
        input_file,
        max_genes=10000,           # Keep at most 10,000 genes
        filter_variance=True,       # Remove low-variance genes
        variance_percentile=70,     # Keep top 30% most variable
        output_path=output_file
    )
    
    print(f"\n✅ Done! Use '{output_file}' in the Streamlit app")


def quick_start_option_2():
    """Option 2: Pre-filter from command line."""
    print("\n" + "="*60)
    print("OPTION 2: Command Line (1-liner)")
    print("="*60)
    
    print("""
    Run this in your terminal (replace filenames):
    
    python large_dataset_utils.py your_large_data.csv your_filtered_data.csv
    
    Then upload 'your_filtered_data.csv' to the Streamlit app.
    """)


def quick_start_option_3():
    """Option 3: Check memory before uploading."""
    print("\n" + "="*60)
    print("OPTION 3: Check Memory Requirements")
    print("="*60)
    
    from large_dataset_utils import LargeDatasetHandler
    
    # Example: Check memory for your dataset
    n_samples = int(input("\nHow many samples in your dataset? (e.g., 100000): "))
    n_genes = int(input("How many genes in your dataset? (e.g., 20000): "))
    
    handler = LargeDatasetHandler()
    memory_mb = handler.estimate_memory(n_samples, n_genes)
    
    print(f"\n📊 Dataset Size Analysis")
    print(f"   Samples: {n_samples:,}")
    print(f"   Genes:  {n_genes:,}")
    print(f"   Estimated Memory: {memory_mb:.1f} MB")
    
    if memory_mb > 2000:
        print(f"\n⚠️  Large dataset detected!")
        print(f"   Recommendation: Pre-filter to ~10,000 genes")
        filtered_mem = handler.estimate_memory(n_samples, 10000)
        print(f"   After filtering: ~{filtered_mem:.1f} MB")
    else:
        print(f"\n✅ Good size! You can upload directly.")


def quick_start_option_4():
    """Option 4: Get recommended parameters."""
    print("\n" + "="*60)
    print("OPTION 4: Get Recommended Settings")
    print("="*60)
    
    n_samples = int(input("\nHow many samples? (e.g., 100000): "))
    n_genes = int(input("How many genes (after filtering)? (e.g., 10000): "))
    
    # Determine size category
    if n_samples < 10000 and n_genes < 5000:
        category = "Small"
        particles = 30
        generations = 30
        target_genes = 150
        batch_size = 32
    elif n_samples < 50000 and n_genes < 15000:
        category = "Medium"
        particles = 25
        generations = 20
        target_genes = 100
        batch_size = 64
    else:
        category = "Large"
        particles = 20
        generations = 15
        target_genes = 75
        batch_size = 128
    
    print(f"\n📈 {category.upper()} Dataset Recommended Settings:\n")
    print(f"   GA-PSO Population Size:    {particles}")
    print(f"   GA-PSO Generations:        {generations}")
    print(f"   Target Gene Count:         {target_genes}")
    print(f"   Training Batch Size:       {batch_size}")
    print(f"\nUse these values in the Streamlit app sidebar")


def show_menu():
    """Display main menu."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  🧬 LARGE DATASET QUICK START GUIDE".ljust(57) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝\n")
    
    print("Choose what you need:\n")
    print("1️⃣  Pre-filter my CSV file (I have Python installed)")
    print("2️⃣  Show me the command line way")
    print("3️⃣  Check memory requirements for my dataset")
    print("4️⃣  Get recommended parameter settings")
    print("5️⃣  Show me the guide to optimization")
    print("0️⃣  Exit\n")


def main():
    """Main menu handler."""
    while True:
        show_menu()
        choice = input("Enter your choice (0-5): ").strip()
        
        if choice == '1':
            try:
                quick_start_option_1()
            except FileNotFoundError:
                print("❌ File not found. Please check the filename.")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '2':
            quick_start_option_2()
        
        elif choice == '3':
            quick_start_option_3()
        
        elif choice == '4':
            quick_start_option_4()
        
        elif choice == '5':
            print("\n📚 Opening LARGE_DATASET_GUIDE.md...")
            print("   Full guide saved in: LARGE_DATASET_GUIDE.md")
        
        elif choice == '0':
            print("\n✅ Goodbye! Good luck with your dataset. 🧬")
            break
        
        else:
            print("❌ Invalid choice. Try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] in ['-1', '--prefilter'] and len(sys.argv) > 3:
            from large_dataset_utils import reduce_dataset_size
            reduce_dataset_size(sys.argv[2], sys.argv[3])
        else:
            print("Usage:")
            print("  python quick_start.py                    # Interactive menu")
            print("  python quick_start.py -1 input.csv output.csv  # Quick prefilter")
    else:
        # Interactive mode
        main()
