'''
Created on 2017年5月21日

@author: weizhen
'''
class PTBInput(object):
    """The input data."""
    
    def __init__(self,config,data,name=None):
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)/batch_size)-1)/num_steps
        self.input_data,self.targets=reader.ptb_producer(data,batch_size,num_steps,name=name)
        