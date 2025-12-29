from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,UEAloader#, Dataset_M4, PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, #, ChannelUnrolledDataset
from .uea import collate_fn
from torch.utils.data import DataLoader
from functools import partial

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTh3': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'UEA': UEAloader,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
}




class CachedIterable:
    def __init__(self, iterable):
        self.iterable = iterable
        self.cache = []
        self._cached = False

    def __iter__(self):
        if self._cached:
            yield from self.cache
        else:
            for item in self.iterable:
                self.cache.append(item)
                yield item
            self._cached = True
            
def data_provider(args, flag):
    Data = data_dict[args.data.dataset]
    print('args.data.dataset: ', args.data.dataset)
    timeenc = 0 if args.data.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.data.batch_size
    freq = args.data.freq

    if args.data.setup == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.data.root_path,
            win_size=args.data.context_length,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.data.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.data.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.data.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
        return data_set, data_loader
    else:
        if args.data.dataset == 'm4':
            drop_last = False
        if args.data.dataset == "custom":
            data_set = Data(
                args = args,
                root_path=args.data.root_path,
                data_path=args.data.data_path,
                flag=flag,
                size=[args.data.context_length, args.data.label_length, args.data.prediction_length],
                features=args.data.features,
                target=args.data.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.train.seasonal_patterns,)
                # channel_group_size=channel_group_size)
        else:
            data_set = Data(
                root_path=args.data.root_path,
                data_path=args.data.data_path,
                flag=flag,
                size=[args.data.context_length, args.data.label_length, args.data.prediction_length],
                features=args.data.features,
                target=args.data.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.train.seasonal_patterns, 
            )
        # if args.data.features == 'M' and flag.lower() == "train":
        #     data_set = ChannelUnrolledDataset(data_set)
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=drop_last)
        # data_loader = CachedIterable(data_loader)
        return data_set, data_loader

