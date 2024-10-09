from hvac.entity.EnvCore import EnvCore

class Env:
    def __init__(self):
        self.n_floors=2
        self.n_homes=4
        self.n_agents=self.n_homes*self.n_floors

        self.floor = [[EnvCore() for j in range(self.n_homes)]for i in range(self.n_floors)]

        self.state_space=[]
        self.action_space=[]
        self.state=[]
        self.next_state = []
        self.reward=[]
        self.done=[]
        self.info=[]
        for i in range(self.n_floors):
            for j in range(self.n_homes):
                self.state_space.append(self.floor[i][j].state_space)
                self.action_space.append(self.floor[i][j].action_space)
                self.state.append(self.floor[i][j].reset())
                self.reward.append(self.floor[i][j].get_reward())
                self.done.append(self.floor[i][j].get_done())
                self.info.append(self.floor[i][j].get_info())
                self.next_state.append(self.floor[i][j].get_state())

    def reset(self):
        state=[]
        for i in range(self.n_floors):
            for j in range(self.n_homes):
                state.append(self.floor[i][j].reset())
        return state

    def step(self, actions):
        n_action=0
        for i in range(self.n_floors):
            for j in range(self.n_homes):
                self.state[n_action], self.reward[n_action], self.done[n_action], self.info[n_action] = self.floor[i][j].step(actions[n_action])
                n_action+=1
        return self.state,self.reward,self.done,self.info

    def get_info(self,agent_i):
        return self.floor[agent_i//self.n_homes][(agent_i)%self.n_homes].get_info()

if __name__ == '__main__':
    for i in range(3):
        print(i,end=" ")
        if i == 0:
            print("first")
        elif i + 1 == 3:
            print("last")
        else:
            print(i)