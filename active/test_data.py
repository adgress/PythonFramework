import active_project_configs as configs_lib
from utility import helper_functions
import sklearn
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
pc = configs_lib.ProjectConfigs()

main_configs = configs_lib.MainConfigs(pc)

data_file = '../' + main_configs.data_file
data_and_splits = helper_functions.load_object(data_file)
learner = main_configs.learner
skl = Ridge(normalize=True)

x = data_and_splits.data.x
y = data_and_splits.data.y

select_k_best = SelectKBest(f_regression, 50)
x = select_k_best.fit_transform(x,y)
skl.fit(x,y)
score = skl.score(x,y)
print 'R2 Score: ' + str(score)
pass