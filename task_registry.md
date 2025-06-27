# PSETAE Task Registry

This document serves as a registry for all tasks created on the MCP server. It helps track and audit the workflow, ensuring all required tasks are created and processed correctly.

## Sentinel-1 Data Extraction Tasks

| Task ID | Title | Description | Assigned To | GeoJSON File | Output Directory | Status | Creation Date |
|---------|-------|-------------|-------------|--------------|------------------|--------|--------------|
| 5008ac3e-dbc7-4390-801b-92922d0c606e | Sentinel-1 Data Extraction 2024-09-01 to 2025-03-31 | Extract Sentinel-1 data for test split | sentinel1-data-extraction-agent | test_new_wgs84_test.geojson | testdirectory | Processing | 2025-06-27 |
| ba53ceee-96de-4a78-b2b9-5b42aa6726a2 | Sentinel-1 Data Extraction 2024-09-01 to 2025-03-31 | Extract Sentinel-1 data for training split | sentinel1-data-extraction-agent | croptype_KA28_wgs84_training_622.geojson | traindirectory | Created | 2025-06-27 |
| fe21fa57-e8e1-468b-b96e-b3b3c907020a | Sentinel-1 Data Extraction 2024-09-01 to 2025-03-31 | Extract Sentinel-1 data for validation split | sentinel1-data-extraction-agent | croptype_KA28_wgs84_validation_622.geojson | validationdirectory | Created | 2025-06-27 |

## Sentinel-1 Tile Coverage Tasks

| Task ID | Title | Description | Assigned To | GeoJSON File | Output File | Status | Creation Date |
|---------|-------|-------------|-------------|--------------|-------------|--------|--------------|
| 181d344c-066c-4f1f-ae19-4b4fcc6c0eb6 | Test Sentinel-1 Tile Coverage | Analyze Sentinel-1 tile coverage for test area | sentinel1-tile-coverage-agent | test_area.geojson | sentinel1_coverage.txt | Completed | 2025-06-26 |

## Model Training Tasks

| Task ID | Title | Description | Assigned To | Input Directory | Output Directory | Status | Creation Date |
|---------|-------|-------------|-------------|----------------|------------------|--------|--------------|
| *No tasks created yet* | | | | | | | |

## Model Inference Tasks

| Task ID | Title | Description | Assigned To | Model Path | Input Data | Output Directory | Status | Creation Date |
|---------|-------|-------------|-------------|-----------|------------|------------------|--------|--------------|
| *No tasks created yet* | | | | | | | | |

## How to Use This Registry

1. **Adding New Tasks**: When creating a new task via the MCP server, add an entry to this registry with all relevant details.
2. **Updating Task Status**: Update the status field as tasks progress (Created → Processing → Completed/Failed).
3. **Workflow Verification**: Before running any workflow, verify that all required tasks exist in this registry and on the MCP server.
4. **Auditing**: Use this registry to audit completed workflows and ensure all required steps were performed.

## Required Tasks for Sentinel-1 Extraction Workflow

The Sentinel-1 extraction workflow requires three separate extraction tasks:
1. **Test Split**: Using test GeoJSON file, writing to testdirectory
2. **Training Split**: Using training GeoJSON file, writing to traindirectory
3. **Validation Split**: Using validation GeoJSON file, writing to validationdirectory

**IMPORTANT**: Always verify all three tasks exist before running the extraction workflow.
