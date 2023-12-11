import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here

    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0 and convert the DataFrame to integers
    car_matrix = car_matrix.fillna(0).astype(int)

    # Set diagonal values to 0
    car_matrix.values[[range(len(car_matrix))]*2] = 0

    return car_matrix

    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))
    

    return dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes
    return list()


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    # Calculate the average value of the 'truck' column for each 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of routes in ascending order
    filtered_routes.sort()

    return filtered_routes
    return list()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
     # Apply the modification logic to the values in the DataFrame
    modified_matrix = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    df['timeDifference'] = df['endTimestamp'] - df['startTimestamp']

    incorrect_timestamps = df.groupby(['id', 'id_2']).apply(lambda group: not all(
        (group['startTimestamp'].min().time() == pd.Timestamp('00:00:00').time()) and
        (group['endTimestamp'].max().time() == pd.Timestamp('23:59:59').time()) and
        (group['timeDifference'].min() >= pd.Timedelta(days=6, hours=23, minutes=59, seconds=59))
        for day in range(7)
    )).droplevel(2)

    return incorrect_timestamps
    return pd.Series()
