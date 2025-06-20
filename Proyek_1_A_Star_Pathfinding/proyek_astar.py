# Bagian 1: Import Library yang Dibutuhkan
import pandas as pd
import numpy as np
import heapq
import math

# Bagian 2: Fungsi Kalkulasi Jarak (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    """Menghitung jarak antara dua titik koordinat menggunakan formula Haversine"""
    R = 6371  # Radius bumi (Km)
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Bagian 3: Implementasi Algoritma A* Search
def a_star_search(city_data, start_city_name, end_city_name):
    """Implementasi algoritma A* untuk mencari rute optimal antar kota"""
    lat_col = None
    lng_col = None
    lat_variations = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
    lng_variations = ['lng', 'lon', 'longitude', 'Longitude', 'LNG', 'LON', 'LONGITUDE']
    for col in city_data.columns:
        if col in lat_variations:
            lat_col = col
        if col in lng_variations:
            lng_col = col
    if not lat_col or not lng_col:
        print(f"Error: Tidak dapat menemukan kolom koordinat dalam dataset.")
        print(f"Kolom yang tersedia: {city_data.columns.tolist()}")
        return None, None
    cities = {}
    for index, row in city_data.iterrows():
        city_name = str(row['city']).strip()
        lat = float(row[lat_col])
        lng = float(row[lng_col])
        cities[city_name] = (lat, lng)
    
    # Inisialisasi Algoritma A*
    open_set = []
    heapq.heappush(open_set, (0, start_city_name))
    came_from = {}
    g_score = {city: float('inf') for city in cities}
    g_score[start_city_name] = 0
    
    # Hitung heuristic
    start_coords = cities[start_city_name]
    end_coords = cities[end_city_name]
    h_score = haversine(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
    
    f_score = {city: float('inf') for city in cities}
    f_score[start_city_name] = h_score

    visited = set()

    while open_set:
        current_f, current_city_name = heapq.heappop(open_set)
        
        if current_city_name in visited:
            continue
            
        visited.add(current_city_name)
        
        # Jika sudah sampai tujuan
        if current_city_name == end_city_name:
            path = []
            total_distance = g_score[end_city_name]
            temp_city = current_city_name
            
            while temp_city in came_from:
                path.append(temp_city)
                temp_city = came_from[temp_city]
            path.append(start_city_name)
            path.reverse()
            
            return path, total_distance

        # Explore semua kota lain
        current_coords = cities[current_city_name]
        
        for neighbor_city_name, neighbor_coords in cities.items():
            if neighbor_city_name == current_city_name or neighbor_city_name in visited:
                continue

            # Hitung jarak ke neighbor
            distance_to_neighbor = haversine(
                current_coords[0], current_coords[1],
                neighbor_coords[0], neighbor_coords[1]
            )
            
            tentative_g_score = g_score[current_city_name] + distance_to_neighbor
            
            if tentative_g_score < g_score[neighbor_city_name]:
                came_from[neighbor_city_name] = current_city_name
                g_score[neighbor_city_name] = tentative_g_score
                
                # Hitung heuristic ke tujuan
                h_score = haversine(
                    neighbor_coords[0], neighbor_coords[1],
                    end_coords[0], end_coords[1]
                )
                
                f_score[neighbor_city_name] = tentative_g_score + h_score
                heapq.heappush(open_set, (f_score[neighbor_city_name], neighbor_city_name))

    return None, None

# Bagian 4: Fungsi Utilitas
def load_and_display_data(file_path):
    """Load data CSV dan tampilkan informasi dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat!")
        print(f"Jumlah total baris: {len(df)}")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        print(f"\nContoh data (5 baris pertama):")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
        print("Pastikan file CSV berada di folder yang sama dengan script Python ini.")
        return None
    except Exception as e:
        print(f"Error saat membaca file: {e}")
        return None

def show_available_cities(city_df, limit=50):
    """Tampilkan daftar kota yang tersedia"""
    cities = city_df['city'].unique()
    print(f"\nDaftar kota yang tersedia ({len(cities)} kota):")
    
    if len(cities) > limit:
        print(f"Menampilkan {limit} kota pertama:")
        for i, city in enumerate(cities[:limit]):
            print(f"{i+1:2d}. {city}")
        print(f"... dan {len(cities) - limit} kota lainnya")
    else:
        for i, city in enumerate(cities):
            print(f"{i+1:2d}. {city}")

def find_similar_cities(city_df, search_term, max_results=5):
    """Cari kota yang mirip dengan input pengguna"""
    cities = city_df['city'].str.lower().str.contains(search_term.lower(), na=False)
    similar_cities = city_df[cities]['city'].tolist()
    return similar_cities[:max_results]

def get_city_input(city_df, prompt_text):
    """Fungsi untuk mendapatkan input kota dari pengguna dengan validasi"""
    available_cities = city_df['city'].unique()
    
    while True:
        city_input = input(prompt_text).strip()
        
        if city_input in available_cities:
            return city_input
        else:
            print(f"Kota '{city_input}' tidak ditemukan dalam dataset.")
            
            # Cari kota yang mirip
            similar_cities = find_similar_cities(city_df, city_input)
            if similar_cities:
                print(f"Mungkin maksud Anda: {', '.join(similar_cities)}")
            
            print("Ketik 'list' untuk melihat semua kota yang tersedia, atau coba lagi:")
            choice = input().strip().lower()
            if choice == 'list':
                show_available_cities(city_df)

# Bagian 5: Blok Eksekusi Utama
if __name__ == "__main__":
    print("="*70)
    print("PROGRAM PENCARIAN RUTE OPTIMAL MENGGUNAKAN ALGORITMA A*")
    print("="*70)
    
    # Load data
    file_path = 'Worldwide Travel Cities Dataset (Ratings and Climate).csv'
    full_city_df = load_and_display_data(file_path)
    
    if full_city_df is None:
        exit()
    
    # Cek kolom yang diperlukan
    required_columns = ['city']
    missing_columns = [col for col in required_columns if col not in full_city_df.columns]
    
    if missing_columns:
        print(f"Error: Kolom yang diperlukan tidak ditemukan: {missing_columns}")
        exit()
    
    # Filter data jika diperlukan (opsional
    print(f"\nApakah Anda ingin memfilter data berdasarkan negara tertentu? (y/n): ", end="")
    filter_choice = input().strip().lower()
    
    if filter_choice == 'y':
        if 'country' in full_city_df.columns:
            available_countries = full_city_df['country'].unique()
            print(f"Negara yang tersedia: {', '.join(available_countries[:10])}...")
            countries_input = input("Masukkan nama negara (pisahkan dengan koma): ").strip()
            selected_countries = [country.strip() for country in countries_input.split(',')]
            
            city_df = full_city_df[full_city_df['country'].isin(selected_countries)].copy()
            city_df = city_df.drop_duplicates(subset='city', keep='first')
            print(f"Data difilter. Tersedia {len(city_df)} kota dari negara yang dipilih.")
        else:
            print("Kolom 'country' tidak tersedia. Menggunakan semua data.")
            city_df = full_city_df.drop_duplicates(subset='city', keep='first')
    else:
        city_df = full_city_df.drop_duplicates(subset='city', keep='first')
        print(f"Menggunakan semua data. Total: {len(city_df)} kota unik.")
    
    # Tampilkan beberapa kota sebagai contoh
    show_available_cities(city_df, limit=20)
    
    print("\n" + "="*50)
    print("PENCARIAN RUTE")
    print("="*50)
    
    start_city = get_city_input(city_df, "Masukkan nama kota awal: ")
    end_city = get_city_input(city_df, "Masukkan nama kota tujuan: ")
    
    if start_city == end_city:
        print("Kota awal dan tujuan sama!")
        exit()
    
    print(f"\nMencari rute optimal dari {start_city} ke {end_city}...")
    print("Menggunakan algoritma A* Search...")
    
    # Jalankan algoritma A*
    path, distance = a_star_search(city_df, start_city, end_city)
    
    print("\n" + "="*50)
    print("HASIL PENCARIAN")
    print("="*50)
    
    if path and distance:
        print("✅ RUTE OPTIMAL DITEMUKAN!")
        print(f"\nRute: {' → '.join(path)}")
        print(f"Total Jarak: {distance:.2f} km")
        print(f"Jumlah kota yang dilalui: {len(path)}")
        
        # Tampilkan detail setiap segmen
        if len(path) > 1:
            print(f"\nDetail perjalanan:")
            total_check = 0
            for i in range(len(path) - 1):
                # Cari koordinat kota
                city1_data = city_df[city_df['city'] == path[i]].iloc[0]
                city2_data = city_df[city_df['city'] == path[i+1]].iloc[0]
                
                # Deteksi kolom koordinat
                lat_col = None
                lng_col = None
                lat_variations = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
                lng_variations = ['lng', 'lon', 'longitude', 'Longitude', 'LNG', 'LON', 'LONGITUDE']
                
                for col in city_df.columns:
                    if col in lat_variations:
                        lat_col = col
                    if col in lng_variations:
                        lng_col = col
                
                if lat_col and lng_col:
                    segment_distance = haversine(
                        city1_data[lat_col], city1_data[lng_col],
                        city2_data[lat_col], city2_data[lng_col]
                    )
                    total_check += segment_distance
                    print(f"  {i+1}. {path[i]} → {path[i+1]}: {segment_distance:.2f} km")
            
            print(f"\nVerifikasi total jarak: {total_check:.2f} km")
    else:
        print("❌ RUTE TIDAK DITEMUKAN")
        print("Kemungkinan penyebab:")
        print("- Tidak ada data koordinat yang valid")
        print("- Algoritma tidak dapat menemukan jalur optimal")