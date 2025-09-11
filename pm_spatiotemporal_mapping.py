# file: pm_spatiotemporal_mapping.py
# Purpose: Generate synthetic spatiotemporal PM2.5/PM10 data and create IDW heatmaps

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def generate_synthetic_spatiotemporal(n_points=500, n_stations=50, start_time=None, hours=24, bbox=None, random_state=42):
    """
    Generate synthetic spatiotemporal PM2.5 and PM10 observations.
    - n_points: total observations (should be >= 100)
    - n_stations: distinct sensor locations
    - start_time: datetime or None (defaults to now)
    - hours: temporal window length (number of hourly timesteps)
    - bbox: (min_lon, max_lon, min_lat, max_lat) for spatial domain; defaults provided
    """
    rng = np.random.default_rng(random_state)
    if bbox is None:
        bbox = (-0.25, 0.10, 51.45, 51.60)  # small area (e.g., part of a city)
    min_lon, max_lon, min_lat, max_lat = bbox

    if start_time is None:
        start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    station_lons = rng.uniform(min_lon, max_lon, size=n_stations)
    station_lats = rng.uniform(min_lat, max_lat, size=n_stations)

    timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
    records = []

    n_sources = 3
    source_lons = rng.uniform(min_lon, max_lon, n_sources)
    source_lats = rng.uniform(min_lat, max_lat, n_sources)
    source_strength_pm25 = rng.uniform(20, 60, n_sources)
    source_strength_pm10 = rng.uniform(30, 80, n_sources)

    for si in range(n_stations):
        lon = station_lons[si]
        lat = station_lats[si]
        baseline25 = rng.uniform(5, 15)
        baseline10 = rng.uniform(10, 25)
        local_factor = rng.normal(1.0, 0.05)

        for ti, ts in enumerate(timestamps):
            hour = ts.hour
            diurnal = 1.0 + 0.5 * np.sin((hour / 24.0) * 2 * np.pi)
            weather_idx = 1.0 + 0.3 * np.sin((ti / hours) * 2 * np.pi + rng.normal(0, 0.2))

            dists = np.sqrt((lon - source_lons)**2 + (lat - source_lats)**2)
            influence25 = np.sum(source_strength_pm25 * np.exp(- (dists / 0.02)**2))
            influence10 = np.sum(source_strength_pm10 * np.exp(- (dists / 0.02)**2))

            wind_speed = abs(rng.normal(3.0, 1.0))  # m/s
            mixing_factor = 1.0 / (1.0 + 0.2 * wind_speed)
            humidity = np.clip(rng.normal(60, 15), 10, 100) / 100.0

            pm25 = baseline25 * local_factor * diurnal * weather_idx                    + influence25 * mixing_factor * (0.8 + 0.4 * humidity)                    + rng.normal(scale=2.0 + 0.05 * influence25)

            pm10 = baseline10 * local_factor * diurnal * weather_idx                   + influence10 * mixing_factor * (0.9 + 0.5 * humidity)                   + rng.normal(scale=3.0 + 0.06 * influence10)

            pm25 = max(0.0, pm25)
            pm10 = max(0.0, pm10)

            records.append({
                "station_id": f"S{si:03d}",
                "lon": lon,
                "lat": lat,
                "timestamp": ts.isoformat(),
                "hour": hour,
                "wind_speed_m_s": round(wind_speed, 2),
                "humidity": round(humidity, 2),
                "pm25_ug_m3": round(pm25, 3),
                "pm10_ug_m3": round(pm10, 3)
            })

    df = pd.DataFrame.from_records(records)

    if df.shape[0] < n_points:
        raise ValueError(f"Generated only {df.shape[0]} rows which is < requested n_points={n_points}. Increase stations/hours.")
    return df

def idw_interpolation(xy, values, xi, yi, power=2, k=8, eps=1e-12):
    """
    Simple IDW interpolation using KDTree nearest neighbors.
    - xy: (N,2) array of sample points (lon,lat)
    - values: (N,) sample values
    - xi, yi: grid coordinates (1D arrays) to interpolate onto
    Returns grid Z of shape (len(yi), len(xi))
    """
    pts = np.column_stack(xy)
    tree = cKDTree(pts)
    XI, YI = np.meshgrid(xi, yi)
    grid_points = np.column_stack([XI.ravel(), YI.ravel()])

    dists, idxs = tree.query(grid_points, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / (dists + eps)**power
    vals = values[idxs]
    numer = np.sum(weights * vals, axis=1)
    denom = np.sum(weights, axis=1)
    Z = numer / denom
    return Z.reshape(YI.shape), XI, YI

def plot_heatmap(lons, lats, values, grid_lon, grid_lat, title, outpath=None, cmap="inferno"):
    plt.figure(figsize=(8,6))
    plt.pcolormesh(grid_lon, grid_lat, values, shading="auto", cmap=cmap)
    plt.scatter(lons, lats, c='white', s=10, edgecolors='k', linewidth=0.3)
    plt.colorbar(label=title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
        plt.close()
    else:
        plt.show()

def main():
    N_POINTS_MIN = 100
    n_stations = 60
    hours = 24
    n_points_requested = 500
    bbox = (-0.25, 0.10, 51.45, 51.60)

    df = generate_synthetic_spatiotemporal(n_points=n_points_requested, n_stations=n_stations, hours=hours, bbox=bbox, random_state=123)
    print("Generated dataset rows:", len(df))
    print(df[["station_id","lon","lat","timestamp","pm25_ug_m3","pm10_ug_m3"]].head())

    os.makedirs("data", exist_ok=True)
    os.makedirs("results/heatmaps_seq", exist_ok=True)

    csv_path = os.path.join("data", "pm_spatiotemporal.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    unique_ts = sorted(df["timestamp"].unique())
    sample_ts = unique_ts[len(unique_ts)//2]
    df_ts = df[df["timestamp"] == sample_ts].copy()
    print("Mapping timestamp:", sample_ts, "observations:", len(df_ts))

    min_lon, max_lon = df["lon"].min(), df["lon"].max()
    min_lat, max_lat = df["lat"].min(), df["lat"].max()
    pad_lon = (max_lon - min_lon) * 0.05
    pad_lat = (max_lat - min_lat) * 0.05
    grid_x = np.linspace(min_lon - pad_lon, max_lon + pad_lon, 120)
    grid_y = np.linspace(min_lat - pad_lat, max_lat + pad_lat, 120)

    xy = df_ts[["lon","lat"]].values
    pm25_vals = df_ts["pm25_ug_m3"].values
    pm10_vals = df_ts["pm10_ug_m3"].values

    Z25, GX, GY = idw_interpolation(xy, pm25_vals, grid_x, grid_y, power=2, k=8)
    Z10, _, _ = idw_interpolation(xy, pm10_vals, grid_x, grid_y, power=2, k=8)

    out25 = f"results/heatmap_pm25_{sample_ts.replace(':','').replace('-','').replace('T','_')}.png"
    out10 = f"results/heatmap_pm10_{sample_ts.replace(':','').replace('-','').replace('T','_')}.png"
    plot_heatmap(df_ts["lon"].values, df_ts["lat"].values, Z25, GX, GY, f"PM2.5 (ug/m3) @ {sample_ts}", outpath=out25)
    plot_heatmap(df_ts["lon"].values, df_ts["lat"].values, Z10, GX, GY, f"PM10 (ug/m3) @ {sample_ts}", outpath=out10)
    print("Saved sample heatmaps:", out25, out10)

    seq_ts = unique_ts[::2]
    for i, ts in enumerate(seq_ts):
        df_t = df[df["timestamp"] == ts]
        xy_t = df_t[["lon","lat"]].values
        Z_t, _, _ = idw_interpolation(xy_t, df_t["pm25_ug_m3"].values, grid_x, grid_y, power=2, k=8)
        outseq = f"results/heatmaps_seq/heatmap_{i:03d}.png"
        plot_heatmap(df_t["lon"].values, df_t["lat"].values, Z_t, GX, GY, f"PM2.5 @ {ts}", outpath=outseq)
    print("Saved sequence heatmaps to results/heatmaps_seq/")

    print("\nPM2.5 stats (all):\n", df["pm25_ug_m3"].describe().round(3))
    print("\nPM10 stats (all):\n", df["pm10_ug_m3"].describe().round(3))

if __name__ == "__main__":
    main()
