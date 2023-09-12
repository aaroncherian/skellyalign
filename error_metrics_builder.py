
class ErrorMetricsBuilder:
    def __init__(self, freemocap_df, qualisys_df):
        self.freemocap_df = freemocap_df
        self.qualisys_df = qualisys_df
        self.results = {}

    