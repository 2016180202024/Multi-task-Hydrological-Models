import os


class PathConfig:
    caravan_path = r'D:\data\Caravan'
    caravan_timeseries_path = os.path.join(caravan_path, r'timeseries\csv')
    caravan_attributes_path = os.path.join(caravan_path, r'attributes')
    caravan_shapefiles_path = os.path.join(caravan_path, r'shapefiles')

    root_path = r'D:\experiment'
    # root_path = r'/home/cas-519/storage-2t/wzl/experiment'

    experiment_path = os.path.join(root_path, 'data')
    origin_path = os.path.join(experiment_path, 'origin_data')
    train_path = os.path.join(experiment_path, 'train_data')

    model_path = os.path.join(root_path, 'model')
    model_conf_path = os.path.join(model_path, 'config')

    shp_combined_path = os.path.join(caravan_shapefiles_path, r'combined.shp')
