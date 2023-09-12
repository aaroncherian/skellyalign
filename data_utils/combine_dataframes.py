def load_and_process_data(freemocap_dataframe, qualisys_dataframe):
    # Extract markers and convert to dataframe
    freemocap_dataframe = extract_and_convert_data(freemocap_3d_data, mediapipe_markers, markers_to_extract)
    qualisys_dataframe = extract_and_convert_data(qualisys_3d_data, qualisys_markers, markers_to_extract)
    
    # Add system labels
    freemocap_dataframe['system'] = 'freemocap'
    qualisys_dataframe['system'] = 'qualisys'
    
    # Combine dataframes
    dataframe_of_3d_data = pd.concat([freemocap_dataframe, qualisys_dataframe], ignore_index=True)
    
    return dataframe_of_3d_data