from hvac.entity.EnvCore import EnvCore

class Env:
    def __init__(self):
        self.n_floors=1
        self.n_homes=1
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

    #Get a temperature list for going up and down stairs :upstairs_temperature、downstairs_temperature
    def get_updownstairs_temperature(self):
        upstairs_temperature=[]
        downstairs_temperature=[]
        for i in range(self.n_floors):
            # self.state[i],self.reward[i],self.done[i],self.info[i]=self.floor[i].step(actions[i].tolist())
            for j in range(self.n_homes):
                if i==0 and i+1==self.n_floors:
                    upstairs_temperature.append(self.floor[i][j].get_temperature())
                    downstairs_temperature.append(self.floor[i][j].get_temperature())
                elif i==0:
                    upstairs_temperature.append(self.floor[i + 1][j].get_temperature())
                    downstairs_temperature.append(self.floor[i][j].get_temperature())
                elif i+1==self.n_floors:
                    upstairs_temperature.append(np.array([self.floor[i][j].get_state()[0] for _ in range(len(self.floor[i][j].get_temperature()))]))
                    downstairs_temperature.append(self.floor[i-1][j].get_temperature())
                else:
                    upstairs_temperature.append(self.floor[i + 1][j].get_temperature())
                    downstairs_temperature.append(self.floor[i-1][j].get_temperature())
        return upstairs_temperature,downstairs_temperature

if __name__ == '__main__':
    for i in range(3):
        print(i,end=" ")
        if i == 0:
            print("first")
        elif i + 1 == 3:
            print("last")
        else:
            print(i)