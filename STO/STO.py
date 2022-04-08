import datetime
from sklearn.cluster import DBSCAN

class STO:
    def __init__(self, timebin_duration, weeks_before_and_after):
        self.timebin_duration = timebin_duration
        self.W = weeks_before_and_after
        self.all_points = []
        self.all_trajectories_indexed = dict()
        self.all_transitions_indexed = dict()
        self.labels_indexed = dict()

        self.date_scaler = {'first_monday': None, 'min_date': 10**10}

    def process_trajectory(self, traj):
        trip = [t[:2] for t in traj]
        date_timestamp, hour_in_seconds = self.median_transition_time(
            traj[0][2], traj[-1][2])
        return {
            'trip': trip,
            'date_timestamp': date_timestamp,
            'hour': hour_in_seconds
        }

    def fit_date_scaler(self, trajectories):
        for t in trajectories:
            timestamp = t['date_timestamp']
            if timestamp < self.date_scaler['min_date']:
                self.date_scaler['min_date'] = timestamp
                date = datetime.datetime.fromtimestamp(timestamp)
                self.date_scaler['first_monday'] = date - datetime.timedelta(
                    days=date.weekday())

    def transform_trajectories_by_date_scaler(self, trajectories):
        for i in range(len(trajectories)):
            day_hour = str(
                datetime.datetime.fromtimestamp(
                    trajectories[i]['date_timestamp']).weekday()
            ) + 'd' + str(trajectories[i]['hour'] //
                          self.timebin_duration) + 'h'
            trajectories[i] = {
                'trip': trajectories[i]['trip'],
                'week_id': (datetime.datetime.fromtimestamp(
                    trajectories[i]['date_timestamp']) -
                            self.date_scaler['first_monday']).days // 7,
                'day_hour': day_hour
            }

        return trajectories

    def process_trajectories_week_and_hour(self):
        all_trajectories = self.create_trajectories()
        processed_trajectories = [
            self.process_trajectory(traj) for traj in all_trajectories
        ]

        trips = [t['trip'] for t in processed_trajectories]
        trips = self.process_trips(trips)
        for i in range(len(processed_trajectories)):
            processed_trajectories[i]['trip'] = trips[i]

        self.fit_date_scaler(processed_trajectories)

        processed_trajectories = self.transform_trajectories_by_date_scaler(processed_trajectories) 

        return processed_trajectories

    def index_trajectories(self):
        processed_trajectories = self.process_trajectories_week_and_hour()
        for t in processed_trajectories:
            week_key = t['week_id']
            if self.all_trajectories_indexed.get(week_key) != None:
                self.all_trajectories_indexed[week_key].append(t)
            else:
                self.all_trajectories_indexed[week_key] = [t]
        for week_key in self.all_trajectories_indexed.keys():
            week_dict = {}
            for t in self.all_trajectories_indexed[week_key]:
                day_hour_key = t['day_hour']
                if week_dict.get(day_hour_key) != None:
                    week_dict[day_hour_key].append(t)
                else:
                    week_dict[day_hour_key] = [t]
            self.all_trajectories_indexed[week_key] = week_dict

    def add_to_counts(self, d, e):
        if d.get(e) != None:
            d[e] += 1
        else:
            d[e] = 1
        return d

    def median_transition_time(self, datetime_str1, datetime_str2):
        datetime1 = datetime.datetime.strptime(
            datetime_str1, '%Y-%m-%d-%H:%M:%S').timestamp()
        datetime2 = datetime.datetime.strptime(
            datetime_str2, '%Y-%m-%d-%H:%M:%S').timestamp()
        mean_timestamp = (datetime2 + datetime1) / 2
        dt_mean = datetime.datetime.fromtimestamp(mean_timestamp)
        return mean_timestamp, 3600 * dt_mean.hour + 60 * dt_mean.minute + dt_mean.second

    def calculate_transitions_with_features(self, trips):
        trips = [t['trip'] for t in trips]
        transitions = dict()
        outgoing = dict()
        incoming = dict()
        for rt in trips:
            for i in range(len(rt) - 1):
                transition = [p[0] for p in rt[i:i + 2]]
                transition_str = ','.join(transition)
                transitions = self.add_to_counts(transitions, transition_str)
                outgoing = self.add_to_counts(outgoing, transition[0])
                incoming = self.add_to_counts(incoming, transition[1])
        transitions_with_features = dict()
        for p in transitions:
            transitions_with_features[p] = {
                'passing':
                transitions.get(p),
                'outgoing_ratio':
                transitions.get(p) / outgoing.get(p.split(',')[0]),
                'incoming_ratio':
                transitions.get(p) / incoming.get(p.split(',')[1])
            }

        return transitions_with_features

    def index_transitions(self):
        for week in self.all_trajectories_indexed.keys():
            self.all_transitions_indexed[week] = dict()
            for day_hour in self.all_trajectories_indexed[week].keys():
                self.all_transitions_indexed[week][
                    day_hour] = self.calculate_transitions_with_features(
                        self.all_trajectories_indexed[week][day_hour])

    def euclidean_diff(self, x, y):
        s = 0.0
        for i in range(len(x)):
            s += (x[i] - y[i])**2
        return s**0.5

    def retrieve_stats(self, data, week, day_hour, transition):
        stats = None
        try:
            stats = data[week][day_hour][transition]
        except:
            stats = None
        return stats

    def add_min_distort(self, transition, week, day_hour):
        feature_arrays = []
        for w in range(1, self.W + 1):
            transition_stats_pos = self.retrieve_stats(
                self.all_transitions_indexed, week + w, day_hour, transition)
            if transition_stats_pos == None:
                transition_stats_pos = {'passing': 0, 'outgoing_ratio': 0, 'incoming_ratio': 0}
            feature_arrays.append([
                transition_stats_pos['passing'],
                transition_stats_pos['outgoing_ratio'],
                transition_stats_pos['incoming_ratio']
            ])
            transition_stats_neg = self.retrieve_stats(
                self.all_transitions_indexed, week - w, day_hour, transition)
            if transition_stats_neg == None:
                transition_stats_neg = {'passing': 0, 'outgoing_ratio': 0, 'incoming_ratio': 0}
            feature_arrays.append([
                transition_stats_neg['passing'],
                transition_stats_neg['outgoing_ratio'],
                transition_stats_neg['incoming_ratio']
            ])

        transition_stats = self.all_transitions_indexed[week][day_hour][
            transition]
        transition_stats = [
            transition_stats['passing'], transition_stats['outgoing_ratio'],
            transition_stats['incoming_ratio']
        ]

        distorts = [
            self.euclidean_diff(arr, transition_stats)
            for arr in feature_arrays
        ]
        min_distort = min(distorts)
        self.all_transitions_indexed[week][day_hour][transition][
            'min_distort'] = min_distort

    def calculate_all_min_distorts(self):
        for week in self.all_transitions_indexed.keys():
            for day_hour in self.all_transitions_indexed[week].keys():
                for transition in self.all_transitions_indexed[week][
                        day_hour].keys():
                    self.add_min_distort(transition, week, day_hour)

    def cluster_groups(self):
        for week in self.all_transitions_indexed.keys():
            for day_hour in self.all_transitions_indexed[week].keys():
                transition_features = []
                for transition in self.all_transitions_indexed[week][
                        day_hour].keys():
                    transition_dict = self.all_transitions_indexed[week][day_hour][transition]
                    transition_list = [transition_dict['passing'], transition_dict['outgoing_ratio'], transition_dict['incoming_ratio']]
                    transition_features.append(
                        [transition] + transition_list)
                dbscan = DBSCAN(eps=0.1, min_samples=5)
                transitions = [t[0] for t in transition_features]
                dbscan.fit([t[1:] for t in transition_features])
                labels = []
                for l in dbscan.labels_:
                    if l >= 0:
                        labels.append(0)
                    else:
                        labels.append(1)

                transitions_with_labels = dict()
                for i in range(len(transitions)):
                    transitions_with_labels[transitions[i]] = labels[i]
                self.labels_indexed[str(week) + 'w' +
                                    day_hour] = transitions_with_labels

    def fit(self):
        self.index_trajectories()
        self.index_transitions()
        self.calculate_all_min_distorts()
        self.cluster_groups()

    def predict(self, X):
        labels = []

        trips = [t['trip'] for t in X]
        trips = self.process_trips_transform(trips)
        for i in range(len(X)):
            X[i]['trip'] = trips[i]

        X = self.transform_trajectories_by_date_scaler(X) 

        for x in X:
            trip = x['trip']
            week = x['week_id']
            day_hour = x['day_hour']
            transitions = []
            for i in range(len(trip) - 1):
                transitions.append(trip[i] + ',' + trip[i + 1])
            transition_labels = [
                self.labels_indexed[str(week) + 'w' + day_hour][tr]
                for tr in transitions
            ]
            if sum(transition_labels) > 0:
                labels.append(1)
            else:
                labels.append(0)
        
        return labels

