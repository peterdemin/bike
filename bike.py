
# coding: utf-8

# In[18]:

import datetime
import csv
import math
import itertools
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter, OrderedDict
from operator import itemgetter

get_ipython().magic(u'matplotlib inline')

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

plt.rcParams['figure.figsize'] = (18.0, 8.0)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True


# In[19]:

header = ('datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count')
sample = ('2011-01-01 00:00:00', 1, 0, 0, 1, 9.84, 14.395, 81, 0, 3, 13, 16)


# In[277]:

def make_ridge():
    return Ridge(
        alpha=0.001,
        normalize=True,
        fit_intercept=True,
    )

class SplittedEstimator(object):
    def __init__(self, estimators, splitter):
        self.estimators = estimators
        self.splitter = splitter

    def fit(self, xs, ys):
        xs['__target__'] = ys
        for est, df in zip(self.estimators, self.splitter(xs)):
            y = df.pop('__target__')
            est.fit(df, y)
        xs.pop('__target__')
        return self

    def predict(self, dataframe):
        ys = []
        for est, df in zip(self.estimators, self.splitter(dataframe)):
            y = pd.DataFrame(est.predict(df), index=df.index)
            ys.append(y)
        return pd.concat(ys, axis=0)


class BikeRentPredictor(object):
    def __init__(self):
        pass

    def fit_from_csv(self, filename):
        return self.fit(pd.DataFrame.from_csv(filename))

    def fit(self, df):
        x, y_casual, y_regular, y_total = self.make_train(df)
        self.e_casual, self.e_regular = self.fit_latent_features(x, y_casual, y_regular)
        x['casual'] = y_casual
        x['registered'] = y_regular
        self.estimator = make_ridge()
        self.estimator.fit(x, y_total)
        return self

    def make_train(self, df):
        df = self.fix_atemp(df)
        y_total = df.pop('count')
        self.add_features(df, ignore_columns=('casual', 'registered'))
        df = self.regularize(df)
        y_casual = df.pop('casual')
        y_regular = df.pop('registered')
        return df, y_casual, y_regular, y_total

    def add_features(self, x, ignore_columns=None):
        x.pop('season')
        x['days_ttl'] = (x.index - datetime.datetime(2010, 1, 1)).days
        self.add_flag_combinations(x, ignore_columns)
        x.pop('weather')

    def add_flag_combinations(self, x, ignore_columns=None):
        nonflags = set(x.columns.values).difference(set(ignore_columns or []).union(set(['weather'])))
        wflags = pd.DataFrame(index=x.index)
        for w in range(1, 5):
            wflags['weather%d' % w] = (x.weather == w)
        for flag, nonflag in itertools.product(sorted(wflags.columns.values), sorted(nonflags)):
            x[flag + '_' + nonflag] = wflags[flag] * x[nonflag]
        hflags = pd.DataFrame(index=x.index)
        for h in range(24):
            hflags['hour%d' % h] = (x.index.hour == h)
        for flag, nonflag in itertools.product(sorted(hflags.columns.values), sorted(nonflags)):
            x[flag + '_' + nonflag] = hflags[flag] * x[nonflag]
        nonflags = set(x.columns.values).difference(set(ignore_columns or []).union(set(['weather'])))
        for flag, nonflag in itertools.product(['holiday', 'workingday'], sorted(nonflags)):
            x[flag + '_' + nonflag] = x[flag] * x[nonflag]

    def fit_latent_features(self, x, *ys):
        return [self.latent_estimator().fit(x, y)
                 for y in ys]

    def latent_estimator(self):
        return SplittedEstimator(
            (make_ridge(), make_ridge()),
            self.split_by_workingday
        )

    def fix_atemp(self, dataframe):
        return dataframe[abs(dataframe.temp - dataframe.atemp) < 10]

    def regularize(self, df, ignore_columns=None):
        ignored = set(ignore_columns or [])
        df_reg = pd.DataFrame(index=df.index)
        for col in df:
            if col not in ignored:
                diff = df[col].max() - df[col].min()
                if diff != 0:
                    df_reg[col] = (df[col] - df[col].min()) / diff
                    continue
            df_reg[col] = df[col]
        assert df.shape == df_reg.shape
        return df_reg

    def split_by_workingday(self, df):
        return df[df.workingday >= 0.5], df[df.workingday < 0.5]

    def predict_on_train(self, filename):
        df = pd.DataFrame.from_csv(filename)
        df.pop('casual')
        df.pop('registered')
        df.pop('count')
        return self.predict_on_dataframe(df)

    def predict(self, filename):
        return self.predict_on_dataframe(pd.DataFrame.from_csv(filename))

    def predict_on_dataframe(self, df):
        self.add_features(df)
        df = self.regularize(df)
        y_casual = np.maximum(0, self.e_casual.predict(df))
        y_regular = np.maximum(0, self.e_regular.predict(df))
        df['casual'] = y_casual
        df['registered'] = y_regular
        return self.fix_prediction(self.estimator.predict(df), df.index)

    def fix_prediction(self, y, index):
        return self.to_int(self.trim(self.attach_index(y, index)))
    
    def trim(self, y):
        return y.apply(functools.partial(np.maximum, 0))

    def to_int(self, y):
        return y.applymap(int)

    def attach_index(self, y, index):
        return pd.DataFrame(y, index=index)
    
    def split_by_day(self, df, day):
        return df[df.index.day <= day], df[df.index.day > day]


rent = BikeRentPredictor()
rent.fit_from_csv('train.csv')
y = rent.predict_on_train('train.csv')
df = pd.DataFrame.from_csv('train.csv')
df['predicted'] = y
rent.add_features(df)
df = rent.regularize(df, ['count', 'predicted'])


# In[278]:

y_test = rent.predict('test.csv')
y_test.columns = ['count']
y_test.to_csv('test.csv.y')


# In[281]:

def analize_regular():
    df = pd.DataFrame.from_csv('train.csv')
    x, y_casual, y_regular, y_total = rent.make_train(df)
    e_casual, e_regular = rent.fit_latent_features(x, y_casual, y_regular)
    y_casual_p = pd.DataFrame(e_casual.predict(x), index=y_casual.index)
    y_regular_p = pd.DataFrame(e_regular.predict(x), index=y_regular.index)
    plt.figure()
    (y_casual - y_casual_p).plot(label='dy')
    plt.legend()
    plt.figure()
    y_regular.plot(label='$y$')
    y_regular_p.plot(label='$y_p$')
    (y_regular - y_regular_p).plot(label='$dy$')
    plt.legend()
    # (y_casual - y_casual_p).plot()
    # y_casual_p.plot()
    y_regular_p[:'2011-01-20'].plot(label='yp')
    y_regular[:'2011-01-20'].plot(label='y')
    # plt.legend()
    (y_regular - y_regular_p)[:'2011-01-20'].plot(label='dy')
    plt.legend()
    e0 = e_regular.estimators[1]
    for coef, name in sorted(zip(e0.coef_, x.columns.values)):
        if 'days_ttl' in name:
            print coef, name
    return locals()

bunch = analize_regular()


# In[282]:

# e0, x = bunch['e0'], bunch['x']
# for coef, name in sorted(zip(e0.coef_, x.columns.values)):
# #     if 'days_ttl' in name:
#         print coef, name
# bunch['y_regular']['2011-01-17'].plot(label='y')
# bunch['y_regular_p']['2011-01-17'].plot(label='y')
for e in bunch['e_regular'].estimators:
    pd.DataFrame(e.predict(bunch['x']['2011-01-17']), index=bunch['x']['2011-01-17'].index).plot()
    for coef, name in sorted(zip(e.coef_, bunch['x'].columns.values)):
        if 'hour' in name:
            print coef, name
# bunch['x']['2011-01-17']
# pd.DataFrame.from_csv('train.csv').describe()


# In[280]:

# plt.figure()
# print df.describe()

# plt.figure()
# df.days_ttl.plot()

plt.figure()
for label in ('count', 'predicted'):
    df[label][:'2011-01-10'].plot()
plt.legend()

plt.figure()
for label in ('count', 'predicted'):
    df[label]['2012-12-10':].plot()
plt.legend()

plt.figure()
df['diff'] = df['count'] - df['predicted']
pd.rolling_max(df['diff'], 24).plot();
pd.rolling_min(df['diff'], 24).plot();


# In[279]:

def kaggle_score(a, b):
    return (np.sum((np.log(a + 1) - np.log(b + 1)) ** 2) / len(a)) ** 0.5

print tabulate([('train',) + scores(df['count'], df['predicted'])], headers=headers)


# In[6]:

def gaussian(x, mu, sig):
    return math.exp(-(x - mu)*(x - mu) / (2.0 * sig * sig))


def hour_distance(a, b):
    return float(a == b)
#     g = gaussian(a, b, 0.3)
#     if abs(g) > 0.001:
#         return g
#     else:
#         return 0.0
#     return math.exp(abs(a - b) - 1)


def month_distance(a, b):
    return gaussian(a, b, 2)
#     return (a - b) ** 2


# [hour_distance(x, 12.0) for x in range(0, 24)]


# In[7]:

def read_csv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=',')
        reader.next()  # skip header
        for row in reader:
            yield row


def read_train():
    return read_csv('train.csv')


def read_test():
    return read_csv('test.csv')


def parse_date(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def enhance_row(row):
    date = parse_date(row[0])
    years_ttl = date.year - 2010
    months_ttl = years_ttl * 12 + date.month
    days_ttl = (date - datetime.datetime(2010, 1, 1)).days
    holiday, workingday = bool(row[2]), bool(row[3])
    date_features = (date.day, years_ttl, months_ttl,
                     days_ttl, date.hour, month_distance(date.month, 7))
    hour_features = tuple([hour_distance(date.hour, h)
                           for h in range(0, 24)])
    return date_features + hour_features + tuple(map(float, row[1:]))


feature_names = ('day', 'years_ttl', 'months_ttl',
    'days_ttl', 'hour', 'month_july',
    'hour0', 'hour1', 'hour2', 'hour3', 'hour4', 'hour5', 'hour6', 'hour7',
    'hour8', 'hour9', 'hour10', 'hour11', 'hour12', 'hour13', 'hour14',
    'hour15', 'hour16', 'hour17', 'hour18', 'hour19', 'hour20',
    'hour21', 'hour22', 'hour23',
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed',
)


def make_train_tuple(row, y_idx=-1):
    parts = enhance_row(row)
    return parts[:-3], parts[y_idx]


def make_multi_column_tuple(row):
    parts = enhance_row(row)
    return tuple([parts[:-3]]) + tuple(parts[-3:])


def make_test_row(row):
    return enhance_row(row)


def make_key(row):
    return row[0]
    
    
def write_prediction(filename, keys, values):
    with open(filename, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(('datetime', 'count'))
        for row in itertools.izip(keys, values):
            writer.writerow(row)


def trim(ys, left=0, right=10000):
    return [min(right, max(left, int(y)))
            for y in ys]


def predict_trimmed(predictor, X, left=0, right=10000):
    if hasattr(predictor, 'predict'):
        return map(trim, predictor.predict(X))
    elif callable(predictor):
        return map(trim, predictor(X))


print make_train_tuple(sample)
print make_multi_column_tuple(sample)
print make_test_row(sample)


# In[8]:

workingday_idx = feature_names.index('workingday')
is_workingday = lambda x, y: int(x[workingday_idx] == 0)


def split_by(xs, ys, predicate):
    result = [[[], []],
              [[], []]]
    for x, y in itertools.izip(xs, ys):
        branch = result[predicate(x, y)]
        branch[0].append(x)
        branch[1].append(y)
    return map(
        np.array,
        [result[0][0], result[1][0],
         result[0][1], result[1][1]]
    )


def split_by_date(xs, ys, day_of_month):
    return split_by(xs, ys, lambda x, y: int(x[0] > day_of_month))


def split_by_workingday(xs, ys):
    return split_by(xs, ys, is_workingday)


Xa, Ya = map(np.array, zip(*map(make_train_tuple, read_train())))
X, Xt, Y, Yt = split_by_date(Xa, Ya, day_of_month=15)
Xo = np.array(map(make_test_row, read_test()))
Ko = map(np.array, map(make_key, read_test()))

print 'total known', Xa.shape, Ya.shape
print 'training   ', X.shape, Y.shape
print 'testing    ', Xt.shape, Yt.shape
print 'submission ', Xo.shape

print [x.shape for x in split_by_workingday(X, Y)]


# In[9]:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge


class LatentFeature(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, *args, **kwargs):
        super(LatentFeature, self).__init__(*args, **kwargs)
        self.estimator = estimator
        self.fit = self.estimator.fit
        self.predict = self.estimator.predict
    
    def transform(self, xs):
        y = self.predict(xs)
        return np.vstack((xs.T, y)).T

    @staticmethod
    def split_matrix(xs, lower_than_index=-1):
        return xs[:,:lower_than_index], xs[:,lower_than_index]


estimator = Ridge(
    alpha=0.00001,
    normalize=True,
    fit_intercept=True,
)
t = LatentFeature(estimator)
nxs, ny = LatentFeature.split_matrix(X)
t.fit(nxs, ny)
x_aug = t.transform(nxs)
print X.shape, Y.shape, x_aug.shape
print X[:5][:,-1]
print x_aug[:5][:,-1]
print X[:,-1].mean()
print x_aug[:,-1].mean()
# dir(LatentFeatures)


# In[ ]:

def check_latent_dimension():
    plan = iter(['working casual', 'holiday casual',
                 'working regular', 'holiday regular',
                 'working total', 'holiday total'])
    x_, y_casual, y_regular, y_total = map(np.array, zip(*map(make_multi_column_tuple, read_train())))
    # print x_.shape, y_casual.shape, y_regular.shape, y_total.shape
    results = OrderedDict()
    for y_ in (y_casual, y_regular, y_total):
        x_working, x_holiday, y_working, y_holiday = split_by_workingday(x_, y_)
        # print x_working.shape, y_working.shape, x_holiday.shape, y_holiday.shape
        for x, y in ((x_working, y_working), (x_holiday, y_holiday)):
            x_train, x_test, y_train, y_test = split_by_date(x, y, day_of_month=15)
            # print x_train.shape, y_train.shape, x_test.shape, y_test.shape
            pipeline = fit_pipeline(x_train, y_train)
            y_train_predicted = predict_trimmed(pipeline, x_train)
            y_test_predicted = predict_trimmed(pipeline, x_test)
            results[plan.next()] = {
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_train_predicted': y_train_predicted,
                'y_test': y_test,
                'y_test_predicted': y_test_predicted,
                'pipeline': pipeline,
            }
    return results


def mold_ensemble(predictors):
    working_casual = predictors['working casual']['pipeline']
    holiday_casual = predictors['holiday casual']['pipeline']
    working_regular = predictors['working regular']['pipeline']
    holiday_regular = predictors['holiday regular']['pipeline']
    def predict(xs):
        axs = augment_features(xs, 1)
        for x in xs:
            if is_workingday(x):
                yield predictors['working']['pipeline'].predict([x])
            else:
                yield predictors['holiday']['pipeline'].predict([x])
    return predict


def show_results(title, data):
    print title
    print tabulate(
        [('train',) + scores(data['y_train'], data['y_train_predicted']),
         ('test',) + scores(data['y_test'], data['y_test_predicted'])],
        headers=headers,
    )
    print
    plot_train(data['x_train'], data['y_train'], data['y_train_predicted'])
    plot_train(data['x_test'], data['y_test'], data['y_test_predicted'])


def print_coefficients(machine):
    poly = machine.get_params()['poly']
    linear = machine.get_params()['linear']
    for coef, powers in sorted(zip(linear.coef_, poly.powers_), key=itemgetter(0)):
        print powers, coef,
        for fname in itertools.compress(feature_names, powers):
            print fname,
        print


results = check_latent_dimension()


# In[ ]:

# list(itertools.starmap(show_results, results.iteritems()))

# for title in ('working casual',):  # 'holiday casual'):
#     show_results(title, results[title])
#     print_coefficients(results[title]['pipeline'])

def augment_features(xs, *machines):
    ys = [machine.predict(xs)
          for machine in machines]
    for parts in zip(xs, *ys):
        yield np.concatenate((parts[0], parts[1:]))

ax_train = np.array(list(augment_features(
    results['working total']['x_train'],
    results['working casual']['pipeline'],
    results['working regular']['pipeline'],
)))
ax_test = np.array(list(augment_features(
    results['working total']['x_test'],
    results['working casual']['pipeline'],
    results['working regular']['pipeline'],
)))

pipeline = fit_pipeline(ax_train, results['working total']['y_train'])
y_train_predicted = predict_trimmed(pipeline, ax_train)
y_test_predicted = predict_trimmed(pipeline, ax_test)
pack = {
    'x_train': ax_train,
    'x_test': ax_test,
    'y_train': results['working total']['y_train'],
    'y_train_predicted': y_train_predicted,
    'y_test': results['working total']['y_test'],
    'y_test_predicted': y_test_predicted,
    'pipeline': pipeline,        
}
show_results('augmented working total', pack)
show_results('working total', results['working total'])
# np.array((xrange(10) for i in xrange(5)))

ax_validation = np.array(list(augment_features(
    Xo,
    results['working casual']['pipeline'],
    results['working regular']['pipeline'],
)))
y_validation = predict_trimmed(pipeline, Xo)
write_prediction('predict.csv', Ko, Yo)


# In[ ]:

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def best_from_grid():
    from sklearn import grid_search
    parameters = {
    #     'PCA__n_components': range(5, 20),
        'linear__normalize': (True, False),
        'poly__degree': (1, 2, 3),
        'poly__interaction_only': (True, False)
    }
    parameters = {
    #     'PCA__n_components': [10],
        'linear__normalize': [True],
        'poly__degree': [3],
        'poly__interaction_only': (True, False)
    }
    grid = grid_search.GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=3,
        n_jobs=3,
    )
    grid.fit(Xa, Ya)
    return grid.best_estimator_


def fit_pipeline(xs, ys):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
        ('linear', RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1),
                           normalize=True, fit_intercept=True)),
    ])
    pipeline.fit(xs, ys)
    return pipeline


machine = fit_pipeline(X, Y)
print("Machine parameters:")
best_parameters = machine.get_params()
for param_name in sorted(best_parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

Y_ = predict_trimmed(machine, X)
Yt_ = predict_trimmed(machine, Xt)
Yo = predict_trimmed(machine, Xo)
write_prediction('predict.csv', Ko, Yo)


# In[239]:

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tabulate import tabulate


def kaggle_score(Y, Y_):
    return (sum([(math.log(y_ + 1) - math.log(y + 1)) ** 2
                  for y, y_ in zip(Y, Y_)]) / len(Y)) ** 0.5


def scores(Y, Yp):
    return (
        explained_variance_score(Y, Yp),
        r2_score(Y, Yp),
        kaggle_score(Y, Yp),
        mean_absolute_error(Y, Yp),
        mean_squared_error(Y, Yp),
    )
headers = ('explained_variance', 'r2', 'kaggle', 'mean_absolute', 'mean_squared')

print tabulate([('train',) + scores(Y, Y_), ('test',) + scores(Yt, Yt_)], headers=headers)


# In[11]:

def plot_column(axis, x, ys, label, color='k'):
    idx = feature_names.index(label)
    column = [y[idx] for y in ys]
    axis.plot(x, column, color=color, label=label)


def plot_train(X, Y, Yp):
    fig, (overview, diff, zoom) = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
#     fig, (overview, zoom, features) = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    x = np.linspace(0, len(Y), len(Y))
    overview.plot(x, Y, 'r', label="Y")
    overview.plot(x, Yp, 'b', label="Yp")
    overview.legend()
    diff.plot(x, Yp - Y, 'r', label="dY")
    diff.legend()
#     zoom_to = 0, 100
#     zoom_to = len(Y)/2, len(Y)/2 + 100
#     zoom_to = len(Y) - 100, len(Y)
    zoom_to = map(int, [len(Y)*0.85, len(Y)*0.85 + 100])
    Ys = Y[zoom_to[0]: zoom_to[1]]
    Yps = Yp[zoom_to[0]: zoom_to[1]]
    xs = np.linspace(0, len(Ys), len(Ys))
    zoom.plot(xs, Ys, 'r', label="Y")
    zoom.plot(xs, Yps, 'b', label="Yp")
    zoom.legend()
#     X_zoomed = X[zoom_to[0]: zoom_to[1]]
#     plot_column(features, xs, X_zoomed, 'holiday', 'r')
#     plot_column(features, xs, X_zoomed, 'workingday', 'g')
#     plot_column(features, xs, X_zoomed, 'weekend', 'b')
#     plot_column(features, xs, X_zoomed, 'weather', 'k')
#     features.legend()


import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    return itertools.imap(scalar_map.to_rgba, itertools.count(1))


def discover(x, y):
    fig, axes = plt.subplots(figsize=(16, 2*len(feature_names)), nrows=len(feature_names))
    idx = np.linspace(0, len(y), len(y))
#     axis.plot(idx, y, 'k')
    cmap = get_cmap(16)
    for f, axis in zip(feature_names, axes):
        if 'hour' not in f:
            axis.plot(idx, y, 'k')
            axis2 = axis.twinx()
            plot_column(axis2, idx, x, f, cmap.next())
            axis2.legend()
#     axis.legend()
    return axis

# fig, axes = plt.subplots(figsize=(16, 8), nrows=2)
# idx = np.linspace(0, len(Yo), len(Yo))
# axes[0].plot(idx, Yo)
# plot_column(axes[1], idx, Xo, "season")

plot_train(X, Y, Y_)
plot_train(Xt, Yt, Yt_)
length = 100000
start = 0
stop = start + length
# discover(Xo[start:stop], Yo[start:stop])
# discover(Xo, Yo)
# # axis = plot_line(Yo[480:490])
# # plot_column(axis, xs, X_zoomed, 'holiday', 'r')
# print tabulate(zip(*[feature_names, Xo[434], Xo[550]]))
# print Ko[480], Ko[490]
# bads = Counter()
# for key, value in itertools.islice(zip(Ko, Yo), 10000):
#     if value > 800:
#         date = parse_date(str(key)).date()
#         bads.update([date])
# pprint(dict(bads))


# In[12]:

poly = machine.get_params()['poly']
linear = machine.get_params()['linear']
for coef, powers in sorted(zip(linear.coef_, poly.powers_), key=itemgetter(0)):
    print powers, coef,
    for fname in itertools.compress(feature_names, powers):
        print fname,
    print


# In[13]:

# tempslice = df.temp[:'2011-01-20']
# tempslice.plot(style='b')
# pd.rolling_mean(tempslice, 24*3).plot(figsize=(18, 10), style='r')

# plt.figure()
# df['count'].plot(figsize=(18, 10), style='k')
# df.registered.plot(figsize=(18, 10), style='b')
# df.casual.plot(figsize=(18, 10), style='r')

# plt.figure()
# for column, color in zip(('casual', 'registered', 'count'), 'rgk'):
#     pd.rolling_sum(df[column][:'2012-01-21'], 24*7).plot(figsize=(18, 10), style=color)


# df['count']['2011-01-18':'2011-02-01']
# print df.index.year - 2010
# print df.index.day
# print df.index.month
# print (df.index - datetime.datetime(2010, 1, 1)).days

plt.figure(figsize=(18, 10))
pd.rolling_sum(df.temp, 24*7).plot()
# df.atemp.plot()
# df.atemp.plot()
# dtemp = (df.temp - df.atemp)
# print dtemp[abs(dtemp) > 10].index.values
# df[abs(df.temp - df.atemp) > 10].plot()
# df.atemp.plot()
# rolling_atemp = pd.rolling_mean(df.atemp, 24*3)
# (rolling_atemp - df.atemp).plot()

# diversity = rolling_atemp - df.atemp
# df[abs(diversity) > 10].describe()
# pd.rolling_mean(df.atemp, 24*3).plot(style='r')
# df.atemp['2012-08-16':'2012-08-20'].plot()
# df.temp['2012-08-16':'2012-08-20'].plot()
# print df.atemp['2012-08-16':'2012-08-19'].values
# july_distance = functools.partial(month_distance, b=7)
# plt.figure()
# pd.Series(df.index.month).apply(july_distance).plot(figsize=(18, 10))

# date_features = (date.day, years_ttl, months_ttl,
#                  days_ttl, date.hour, month_distance(date.month, 7))
# hour_features = tuple([hour_distance(date.hour, h)
#                        for h in range(0, 24)])


# In[14]:

hours = pd.date_range(df.index[0], df.index[-1], freq='H')
print hours.shape
print df.shape
dfd = pd.DataFrame(df, index=hours)
pd.rolling_mean(dfd.atemp.interpolate(), 24 * 21).plot(label=r'$atemp_{roll}$')
pd.rolling_mean(dfd.temp.interpolate(), 24 * 21).plot(label='$temp_{roll}$')
pd.stats.moments.ewma(dfd.atemp.interpolate(), com=24*21).plot(label='$ewma$')
dfd.atemp.plot(label='$atemp$')
plt.legend()
# dfd.atemp


# In[15]:

dff = df[abs(df.temp - df.atemp) < 10]


# In[16]:

dff.atemp.plot()


# In[17]:

df = pd.DataFrame.from_csv('train.csv')
# df[df.workingday.bool == 1], df[not df.workingday == 1]
for x in df:
    dx = df[x].max() - df[x].min()
    if dx != 0:
        df[x] = (df[x] - df[x].min()) / dx
df.mean()


# In[ ]:



