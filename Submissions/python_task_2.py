import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    
    G = nx.from_pandas_edgelist(df5, 'id_1', 'id_2', ['distance'])
    distance_matrix = pd.DataFrame(0, index=df5['id_1'].unique(), columns=df5['id_1'].unique())

    for node1 in distance_matrix.index:
        for node2 in distance_matrix.columns:
            if node1 != node2:
                try:
                    # Calculate the shortest path distance
                    distance = nx.shortest_path_length(G, source=node1, target=node2, weight='distance')
                    distance_matrix.at[node1, node2] = distance
                    distance_matrix.at[node2, node1] = distance  # Ensure symmetry
                except nx.NetworkXNoPath:
                    # If no path is found, leave the distance as 0
                    pass

    return distance_matrix
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    upper_triangle = distance_matrix.where(pd.notna(distance_matrix), 0).values
    upper_triangle[np.tril_indices_from(upper_triangle)] = 0
    row_indices, col_indices = np.where(upper_triangle > 0)

    unrolled_df = pd.DataFrame({
        'id_start': distance_matrix.index[row_indices],
        'id_end': distance_matrix.columns[col_indices],
        'distance': upper_triangle[row_indices, col_indices]
    })

    return unrolled_df

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    
    reference_rows = df[df['id_start'] == reference_value]
    reference_avg_distance = reference_rows['distance'].mean()
    threshold_lower = reference_avg_distance - 0.1 * reference_avg_distance
    threshold_upper = reference_avg_distance + 0.1 * reference_avg_distance
    within_threshold = df[(df['distance'] >= threshold_lower) & (df['distance'] <= threshold_upper)]
    result_ids = within_threshold['id_start'].unique()
    result_ids.sort()

    return result_ids
    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    def calculate_time_based_toll_rates(df):
   
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))

    df['start_day'] = df['startTimestamp'].dt.day_name()
    df['end_day'] = df['endTimestamp'].dt.day_name()
    df['start_time'] = df['startTimestamp'].dt.time
    df['end_time'] = df['endTimestamp'].dt.time

    for time_range in weekday_time_ranges:
        mask = (df['start_time'] >= time_range[0]) & (df['end_time'] <= time_range[1]) & \
               (df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.8

    mask = df['start_day'].isin(['Saturday', 'Sunday'])
    df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= 0.7
    return df
