import numpy as np

"""






     kernel_initializer='random_uniform'
     bias_initializer=initializers.Constant(0.1)

def ppo_loss(oldpolicy_probs, advantages, returns, values):
    def loss(y_true, y_pred):
        newpolicy_probs = K.sum(y_true * y_pred,axis=1)
        old_probs = K.sum(y_true * oldpolicy_probs,axis=1)
        
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(old_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(returns - values))
        entropy_loss =  K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta *entropy_loss
        return total_loss
    return loss



states  bs,1,504
prob    bs,1,64
adv     bs,1,1
rwd     bs,1,1
val     bs,1,1
ohot    bs,1,64
rtn     bs,1,1

test_loss(oldprobs, avd, rtn, val, y_t, y_p)

"""
np.random.seed(0)

bs=5
ns=10
na=3
state = np.random.randint(0,2,size=(bs,ns))
oldprobs =  np.random.uniform(0.0,1.0,(bs,na))
y_p = np.random.uniform(0.0,1.0,(bs,na))


avd = np.random.rand(bs)
rtn = np.random.rand(bs)
val = np.random.rand(bs)


y_t=np.zeros((bs,na))
for i in range(bs):
    indx = np.random.randint(0,na)
    y_t[i,indx]=1





def test_loss(oldpolicy_probs, advantages, rewards, values, y_true, y_pred):
    """ Function to try and understand what PPO loss function should be doing
    sub K.'s with np.'s and print results to get a handle on what is going on ....
    # oldpolicy is normalized probs, 
    y_true is one_hot, 
    y_pred is new probs from model
    """
    clipping_val = 0.2
    critic_discount = 0.5
    entropy_beta = 0.001

    #y_true=np.reshape(y_true,(1,1,64))

    newpolicy_probs = np.sum(y_true * y_pred,axis=1)
    print("newpolicy_probs = np.sum(y_true * y_pred,axis=1)\n\n\n",newpolicy_probs.shape,newpolicy_probs)

    old_probs = np.sum(y_true * oldpolicy_probs,axis=1)
    print("old_probs = np.sum(y_true * oldpolicy_probs,axis=1)\n\n\n", old_probs.shape,old_probs)

    
    ratio = np.exp(np.log(newpolicy_probs + 1e-10) - np.log(old_probs + 1e-10))
    print("ratio = np.exp(np.log(newpolicy_probs + 1e-10) - np.log(old_probs + 1e-10))\n\n\n",ratio.shape,ratio)
    
    p1 = ratio * advantages
    print("p1 = ratio * advantages\n\n\n",p1.shape,p1)

    
    p2 = np.clip(ratio, 1 - clipping_val, 1 + clipping_val) * advantages
    print("p2 = np.clip(ratio, 1 - clipping_val, 1 + clipping_val) * advantages\n\n\n",p2.shape,p2)
    
    
    actor_loss = -np.mean(np.minimum(p1, p2))
    print("actor_loss = -np.mean(np.minimum(p1, p2))\n\n\n",actor_loss.shape,actor_loss)
    
    critic_loss = np.mean(np.square(rewards - values))
    print("critic_loss = np.mean(np.square(rewards - values))\n\n\n",critic_loss.shape,critic_loss)
    
    entropy_loss = np.mean(-(newpolicy_probs * np.log(newpolicy_probs + 1e-10)))
    print("entropy_loss = np.mean(-(newpolicy_probs * np.log(newpolicy_probs + 1e-10)))\n\n\n", entropy_loss.shape, entropy_loss)

    
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta *entropy_loss
    print(" total_loss = critic_discount * critic_loss + actor_loss - entropy_beta *entropy_loss\n\n\n",total_loss.shape,total_loss)

    return total_loss



