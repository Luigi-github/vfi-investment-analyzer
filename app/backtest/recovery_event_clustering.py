import pandas as pd


class RecoveryEventClusterer:

    def __init__(
        self,
        minimum_separation_weeks: int = 8
    ):

        if minimum_separation_weeks < 1:
            raise ValueError(
                "minimum_separation_weeks must be >= 1"
            )

        self.minimum_separation_weeks = (
            minimum_separation_weeks
        )


    def detect_events(
        self,
        data
    ):

        required_columns = [

            "VFI",

            "VFI_Cycle_Stage",

            "Close",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        events = (
            data[
                data["VFI_Cycle_Stage"]
                .eq("EARLY_RECOVERY")
            ]
            .copy()
        )

        return events


    def cluster_events_causal(
        self,
        events
    ):

        if events.empty:

            return events.copy()

        events = (
            events
            .sort_index()
            .copy()
        )

        cluster_ids = []

        current_cluster = 0

        previous_date = None

        minimum_gap = pd.Timedelta(
            weeks=self.minimum_separation_weeks
        )

        for current_date in events.index:

            if previous_date is None:

                current_cluster = 1

            else:

                difference = (
                    current_date
                    - previous_date
                )

                if difference >= minimum_gap:

                    current_cluster += 1

            cluster_ids.append(
                current_cluster
            )

            previous_date = current_date

        events[
            "Recovery_Cluster_ID"
        ] = cluster_ids

        events[
            "Weeks_From_Previous_Recovery"
        ] = (
            events.index
            .to_series()
            .diff()
            .dt.days
            / 7.0
        )

        events[
            "Recovery_Cluster_Size"
        ] = (
            events
            .groupby(
                "Recovery_Cluster_ID"
            )["VFI"]
            .transform("size")
        )

        events[
            "Recovery_Events_Seen"
        ] = (
            events
            .groupby(
                "Recovery_Cluster_ID"
            ).cumcount()
            + 1
        )

        return events


    def cluster_events(
        self,
        events
    ):
        """
        Retrospective clustering retained for exploratory analysis.

        Cluster membership is assigned from the complete event history.
        Use cluster_events_causal() for point-in-time signal generation.
        """

        return self.cluster_events_causal(events)


    def select_causal_representative_events(
        self,
        clustered_events
    ):
        """
        Select the first observed event from each causal cluster.

        Unlike the historical representative selector, this never searches
        later events for a lower VFI value.
        """

        if clustered_events.empty:
            return clustered_events.copy()

        representatives = []

        for cluster_id, group in clustered_events.groupby(
            "Recovery_Cluster_ID"
        ):
            group = group.sort_index()
            representative = group.iloc[0].copy()
            representative["Recovery_Cluster_ID"] = cluster_id
            representative["Representative_Method"] = (
                "CAUSAL_FIRST_RECOVERY"
            )
            representative["Representative_Causal"] = True
            representatives.append(representative)

        result = pd.DataFrame(representatives)
        result.index.name = clustered_events.index.name

        return result.sort_index()


    def analyze_causal(
        self,
        data
    ):
        """
        Point-in-time recovery analysis.

        Events are detected and clustered using only their chronological
        ordering. The causal representative is always the first observed
        event in each cluster.
        """

        events = self.detect_events(data)
        clustered_events = self.cluster_events_causal(events)
        representatives = self.select_causal_representative_events(
            clustered_events
        )

        return {
            "events": events,
            "clustered_events": clustered_events,
            "representatives": representatives,
        }


    def cluster_summary(
        self,
        clustered_events
    ):

        if clustered_events.empty:

            return pd.DataFrame()

        rows = []

        for (
            cluster_id,
            group
        ) in clustered_events.groupby(
            "Recovery_Cluster_ID"
        ):

            rows.append({

                "Recovery_Cluster_ID":
                    cluster_id,

                "Start_Date":
                    group.index.min(),

                "End_Date":
                    group.index.max(),

                "Events":
                    len(group),

                "Minimum_VFI":
                    group["VFI"].min(),

                "Maximum_VFI":
                    group["VFI"].max(),

                "Average_VFI":
                    group["VFI"].mean(),
            })

        return pd.DataFrame(rows)


    def select_representative_events(
        self,
        clustered_events
    ):

        if clustered_events.empty:

            return clustered_events.copy()

        representatives = []

        for (
            cluster_id,
            group
        ) in clustered_events.groupby(
            "Recovery_Cluster_ID"
        ):

            group = (
                group
                .sort_index()
            )

            # Seleccionamos el punto donde
            # el VFI alcanza su nivel más bajo
            # dentro del cluster.

            representative_index = (
                group["VFI"]
                .idxmin()
            )

            representatives.append(
                group.loc[
                    representative_index
                ]
            )

        result = pd.DataFrame(
            representatives
        )

        result.index.name = (
            clustered_events.index.name
        )

        return result.sort_index()


    def analyze(
        self,
        data
    ):

        events = (
            self.detect_events(data)
        )

        clustered_events = (
            self.cluster_events(events)
        )

        summary = (
            self.cluster_summary(
                clustered_events
            )
        )

        representatives = (
            self.select_representative_events(
                clustered_events
            )
        )

        return {

            "events":
                events,

            "clustered_events":
                clustered_events,

            "cluster_summary":
                summary,

            "representatives":
                representatives,
        }