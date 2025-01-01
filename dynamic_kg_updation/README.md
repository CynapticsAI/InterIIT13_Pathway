
# Dynamic Graph Update with PATHWAY

This component of the project integrates the **fast and dynamic data engine of PATHWAY** to dynamically update our knowledge graph, which is utilized in **HybridFL**.  

---

## Overview

- **Dynamic Updates**: Incorporates the PATHWAY engine to process input data dynamically.  
- **Flexible Input Handling**: While this implementation works with text files for testing, the UI also supports processing PDFs.  
- **Graph Construction**: The input data is chunked and structured as nodes and edges in the knowledge graph.  

---

## Usage Instructions

Follow the steps below to test the functionality:  

1. **Run the PATHWAY Dynamic Update Script**:  
   This script processes the input data and updates the knowledge graph dynamically.  
   ```bash
   python dynamic_kg_updation\Pathway_Dynamic_Updation.py
   ```

2. **Run the DynamicRAG Script**:  
   This script utilizes the updated graph for further operations.  
   ```bash
   python dynamic_kg_updation\DynamicRAG.py
   ```

---

## Notes

- **File Support**:  
  - For testing, only text files are processed.  
  - The full UI implementation allows for PDF inputs as well.  
- Ensure all required dependencies for the PATHWAY engine are installed before running the scripts.  

---
