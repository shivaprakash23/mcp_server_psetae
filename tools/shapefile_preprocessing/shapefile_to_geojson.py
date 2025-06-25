import geopandas as gpd
import os
import json
import sys
import argparse

def convert_3d_to_2d_coordinates(geojson_data):
    """Convert 3D coordinates to 2D by removing Z values"""
    features = geojson_data['features']
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            # For each polygon, convert coordinates to 2D
            for ring in feature['geometry']['coordinates']:
                for i in range(len(ring)):
                    # Keep only X and Y coordinates
                    ring[i] = ring[i][:2]
        elif feature['geometry']['type'] == 'MultiPolygon':
            # For each multipolygon, convert coordinates to 2D
            for polygon in feature['geometry']['coordinates']:
                for ring in polygon:
                    for i in range(len(ring)):
                        # Keep only X and Y coordinates
                        ring[i] = ring[i][:2]
    return geojson_data

def convert_shapefile(shp_path, output_path):
    """
    Convert shapefile to GeoJSON and save it, removing any 3D components.
    
    Args:
        shp_path (str): Path to the input shapefile
        output_path (str): Path where to save the GeoJSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Reading shapefile from: {shp_path}")
    
    # Read the shapefile
    gdf = gpd.read_file(shp_path)
    
    # Ensure the CRS is WGS84 (EPSG:4326)
    if gdf.crs != 'EPSG:4326':
        print("Converting CRS to WGS84 (EPSG:4326)...")
        gdf = gdf.to_crs('EPSG:4326')
    
    # First save to temporary GeoJSON
    temp_geojson = output_path + '.temp'
    gdf.to_file(temp_geojson, driver='GeoJSON')
    
    # Read the temporary file
    with open(temp_geojson, 'r') as f:
        geojson_data = json.load(f)
    
    # Convert 3D to 2D if needed
    print("Converting 3D coordinates to 2D...")
    geojson_data = convert_3d_to_2d_coordinates(geojson_data)
    
    # Save the modified GeoJSON
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f)
    
    # Remove temporary file
    if os.path.exists(temp_geojson):
        os.remove(temp_geojson)
    
    print(f"GeoJSON saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shapefile_to_geojson.py <input_shapefile_path> <output_geojson_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_shapefile(input_path, output_path)
