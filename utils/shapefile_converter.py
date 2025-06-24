#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shapefile to GeoJSON Converter
This utility converts ESRI Shapefiles (.shp) to GeoJSON format for use with Google Earth Engine.
"""

import os
import json
import argparse
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm

def convert_shapefile_to_geojson(shapefile_path, output_path=None, id_field=None):
    """
    Convert a shapefile to GeoJSON format
    
    Args:
        shapefile_path (str): Path to the input shapefile
        output_path (str, optional): Path for the output GeoJSON file. If None, uses the same name as input with .geojson extension
        id_field (str, optional): Field to use as the feature ID in the GeoJSON
        
    Returns:
        str: Path to the created GeoJSON file
    """
    # Validate input file
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = str(Path(shapefile_path).with_suffix('.geojson'))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Converting {shapefile_path} to GeoJSON...")
    
    # Read shapefile using geopandas
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        raise RuntimeError(f"Error reading shapefile: {str(e)}")
    
    # Ensure the CRS is WGS84 (EPSG:4326) as required by GEE
    if gdf.crs is None:
        print("Warning: Shapefile has no CRS defined. Assuming WGS84.")
        gdf.crs = "EPSG:4326"
    elif gdf.crs != "EPSG:4326":
        print(f"Reprojecting from {gdf.crs} to WGS84 (EPSG:4326)...")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Set the ID field if provided
    if id_field is not None:
        if id_field in gdf.columns:
            # Convert ID field values to strings to ensure compatibility
            gdf[id_field] = gdf[id_field].astype(str)
        else:
            print(f"Warning: ID field '{id_field}' not found in shapefile. Using default index.")
    
    # Convert to GeoJSON
    try:
        # Use to_file for direct conversion
        gdf.to_file(output_path, driver="GeoJSON")
        
        # Optionally, if you need to customize the GeoJSON:
        # geojson_dict = json.loads(gdf.to_json())
        # with open(output_path, 'w') as f:
        #     json.dump(geojson_dict, f)
        
        print(f"Successfully converted to: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error writing GeoJSON: {str(e)}")

def batch_convert(input_dir, output_dir=None, id_field=None, recursive=False):
    """
    Convert all shapefiles in a directory to GeoJSON
    
    Args:
        input_dir (str): Directory containing shapefiles
        output_dir (str, optional): Directory for output GeoJSON files
        id_field (str, optional): Field to use as the feature ID in all GeoJSONs
        recursive (bool): Whether to search subdirectories recursively
        
    Returns:
        list: Paths to all created GeoJSON files
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # Create output directory if specified and doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all shapefiles
    shapefile_paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.shp'):
                    shapefile_paths.append(os.path.join(root, file))
    else:
        shapefile_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                          if f.lower().endswith('.shp') and os.path.isfile(os.path.join(input_dir, f))]
    
    if not shapefile_paths:
        print(f"No shapefiles found in {input_dir}")
        return []
    
    # Convert each shapefile
    geojson_paths = []
    for shp_path in tqdm(shapefile_paths, desc="Converting shapefiles"):
        # Determine output path
        if output_dir is not None:
            rel_path = os.path.relpath(shp_path, input_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.geojson')
            # Ensure subdirectory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            out_path = None  # Will use same location as input
        
        try:
            geojson_path = convert_shapefile_to_geojson(shp_path, out_path, id_field)
            geojson_paths.append(geojson_path)
        except Exception as e:
            print(f"Error converting {shp_path}: {str(e)}")
    
    return geojson_paths

def main():
    """Command line interface for the shapefile converter"""
    parser = argparse.ArgumentParser(description='Convert shapefiles to GeoJSON format for Google Earth Engine')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Conversion mode')
    
    # Single file conversion
    single_parser = subparsers.add_parser('single', help='Convert a single shapefile')
    single_parser.add_argument('--input', '-i', required=True, help='Input shapefile path')
    single_parser.add_argument('--output', '-o', help='Output GeoJSON path')
    single_parser.add_argument('--id-field', help='Field to use as feature ID')
    
    # Batch conversion
    batch_parser = subparsers.add_parser('batch', help='Convert multiple shapefiles')
    batch_parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing shapefiles')
    batch_parser.add_argument('--output-dir', '-o', help='Output directory for GeoJSON files')
    batch_parser.add_argument('--id-field', help='Field to use as feature ID for all files')
    batch_parser.add_argument('--recursive', '-r', action='store_true', help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            convert_shapefile_to_geojson(args.input, args.output, args.id_field)
        elif args.mode == 'batch':
            geojson_paths = batch_convert(args.input_dir, args.output_dir, args.id_field, args.recursive)
            print(f"Successfully converted {len(geojson_paths)} shapefiles to GeoJSON")
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
