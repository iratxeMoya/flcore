#import flcore.models.logistic_regression.server as logistic_regression_server
#import flcore.models.logistic_regression.server as logistic_regression_server
import flcore.models.xgb.server as xgb_server
import flcore.models.random_forest.server as random_forest_server
import flcore.models.linear_models.server as linear_models_server
import flcore.models.weighted_random_forest.server as weighted_random_forest_server
import flcore.models.nn.server as nn_server
import flcore.models.cox.server as cox_server
import flcore.models.rsf.server as rsf_server
import flcore.models.gbs.server as gbs_server
import flcore.models.xgb_irh.server as xgb_irh_server

def get_model_server_and_strategy(config):
    model = config["model"]

    if model in ("logistic_regression", "elastic_net", "lsvc"):
        server, strategy = linear_models_server.get_server_and_strategy(config)
    elif model == "random_forest":
        server, strategy = random_forest_server.get_server_and_strategy(config)
    elif model == "weighted_random_forest":
        server, strategy = weighted_random_forest_server.get_server_and_strategy(config)
    elif model == "xgb":
        server, strategy = xgb_server.get_server_and_strategy(config) #, data)
    elif model == "nn":
        server, strategy = nn_server.get_server_and_strategy(config)
    elif model == "cox":
        server, strategy = cox_server.get_server_and_strategy(config)
    elif model == "rsf":
        server, strategy = rsf_server.get_server_and_strategy(config)
    elif model == "gbs":
        server, strategy = gbs_server.get_server_and_strategy(config)
    elif model == "xgb_irh":
        server, strategy = xgb_irh_server.get_server_and_strategy(config)
    else:
        raise ValueError(f"Unknown model: {model}")

    return server, strategy
