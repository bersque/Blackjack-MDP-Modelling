import numpy as np
import time

player_sums = np.arange(12,22)
dealer_cards = np.arange(2,12)
usable_aces = [0,1]

states = [(p,d,a) for p in player_sums for d in dealer_cards for a in usable_aces]
terminal_states = ['win','loss','draw','bust']
all_states = states + terminal_states
state_idx = {s:i for i,s in enumerate(all_states)}

actions = ['hit','stick']
gamma = 1.0
theta = 1e-6

card_probs = {1:1/13, 2:1/13, 3:1/13, 4:1/13, 5:1/13, 6:1/13,
              7:1/13, 8:1/13, 9:1/13, 10:4/13}


def step_hit(state, card):
    player_sum, dealer_card, usable_ace = state
    if card == 1:
        if player_sum + 11 <= 21:
            return (player_sum+11, dealer_card, 1)
        else:
            return (player_sum+1, dealer_card, usable_ace)
    else:
        new_sum = player_sum + card
        new_usable_ace = usable_ace
        if new_sum > 21 and usable_ace==1:
            new_sum -= 10
            new_usable_ace = 0
        return (new_sum, dealer_card, new_usable_ace)


def dealer_probs(dealer_sum, usable_ace):
    if dealer_sum >= 17:
        return {dealer_sum: 1.0}
    total_probs = {}
    for card, p in card_probs.items():
        if card==1:
            new_sum = dealer_sum + 11
            new_usable_ace = usable_ace + 1
        else:
            new_sum = dealer_sum + card
            new_usable_ace = usable_ace
        while new_sum > 21 and new_usable_ace > 0:
            new_sum -= 10
            new_usable_ace -= 1
        sub_probs = dealer_probs(new_sum, new_usable_ace)
        for k,v in sub_probs.items():
            total_probs[k] = total_probs.get(k,0) + p*v
    return total_probs


def stick_transitions(player_sum, dealer_card):
    dealer_distribution = dealer_probs(dealer_card, 1 if dealer_card==1 else 0)
    transitions = {}
    for d_sum, p in dealer_distribution.items():
        if d_sum > 21 or d_sum < player_sum:
            transitions['win'] = transitions.get('win',0) + p
        elif d_sum > player_sum:
            transitions['loss'] = transitions.get('loss',0) + p
        else:
            transitions['draw'] = transitions.get('draw',0) + p
    return transitions

###macierze P i R
N = len(all_states)
P = np.zeros((N,N,len(actions)))
R = np.zeros((N,N,len(actions)))

for s in states:
    si = state_idx[s]
    player_sum, dealer_card, usable_ace = s
    for a_idx, a in enumerate(actions):
        if a=='hit':
            for card, p in card_probs.items():
                new_state = step_hit(s, card)
                if new_state[0]>21:
                    sj = state_idx['bust']
                    P[sj,si,a_idx] += p
                    R[sj,si,a_idx] = -1
                else:
                    sj = state_idx[new_state]
                    P[sj,si,a_idx] += p
                    R[sj,si,a_idx] = 0
        else:  # stick
            trans = stick_transitions(player_sum, dealer_card)
            for outcome, prob in trans.items():
                sj = state_idx[outcome]
                P[sj,si,a_idx] += prob
                R[sj,si,a_idx] = 1 if outcome=='win' else (-1 if outcome=='loss' else 0)

# terminalne stany absorbujące
for t in terminal_states:
    ti = state_idx[t]
    for a_idx in range(len(actions)):
        P[ti,ti,a_idx] = 1
        R[ti,ti,a_idx] = 0

## value iteration
V = np.zeros(N)
policy = np.zeros(N, dtype=int)
delta_list = 0
while True:
    delta = 0
    for s in states:
        si = state_idx[s]
        v = V[si]
        action_values = np.zeros(len(actions))
        for a_idx in range(len(actions)):
            action_values[a_idx] = np.sum(P[:,si,a_idx] * (R[:,si,a_idx] + gamma*V))
        V[si] = np.max(action_values)
        policy[si] = np.argmax(action_values)
        delta = max(delta, abs(v - V[si]))
        delta_list += 1
    if delta < theta:
        break

# terminalne stany nie mają akcji
for t in terminal_states:
    ti = state_idx[t]
    policy[ti] = -1

##
slownik_co_zrobic = {}
def print_policy(usable_ace_flag):
    for p in player_sums:
        for d in dealer_cards:
            s = (p,d,usable_ace_flag)
            slownik_co_zrobic[s] = actions[policy[state_idx[s]]][0].upper()
print_policy(0)
print_policy(1)

def bj_dobranie(): ### losuje karte
    elements = [1,2,3,4,5,6,7,8,9,10]
    probabilities = [1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 4/13]
    return (((np.random.choice(elements, p=probabilities))))

def generator_stanu_poczatkowego():
    s0 = (0,0,0)
    while s0[0] < 12:
        s0 = step_hit(s0,bj_dobranie())
    gracz_poczatek = (s0[0])
    ace_poczatek = (s0[2])
    krupier_poczatek = (step_hit((0,0,0),bj_dobranie())[0])

    return (gracz_poczatek,krupier_poczatek,ace_poczatek)


def bj():
    s0 = generator_stanu_poczatkowego()
    while True:
        if s0[0] > 21:
            return(-1)
        elif slownik_co_zrobic[s0] == 'H':
            s0 = step_hit(s0,bj_dobranie())
        elif slownik_co_zrobic[s0] == 'S':
            s_k = (s0[1],0,0)
            while s_k[0] < 17:
                s_k = step_hit(s_k,bj_dobranie())
            if s_k[0] > 21:
                return(1)
            krupier_wynik = s_k[0]
            if krupier_wynik > s0[0]:
                return(-1)
            elif krupier_wynik < s0[0]:
                return(1)
            else:
                return(0)

def bj_manual():
    # print("Witaj w mini-grze blackjack! :3 ")
    # print("W każdej turze wybierz akcję: H-dobranie,S-pasowanie.")
    # print("################################################################")
    s0 = generator_stanu_poczatkowego()
    while True:
        print(f"Stan: {s0}")
        if s0[0] > 21:
            return(-1)

        akcja = input("Akcja: ")

        if akcja == "H":
            s0 = step_hit(s0,bj_dobranie())
            print("Dobieranie... ")
            time.sleep(1)

        elif akcja == "S":
            s_k = (s0[1],0,0)
            while s_k[0] < 17:
                s_k = step_hit(s_k,bj_dobranie())
                print(f"Krupier dobiera... {(s0[0],s_k[0],s0[2])}")
                time.sleep(1)
            if s_k[0] > 21:
                return(1)
                break
            krupier_wynik = s_k[0]
            if krupier_wynik > s0[0]:
                return(-1)
                break
            elif krupier_wynik < s0[0]:
                return(1)
                break
            else:
                return(0)
                break
        elif akcja == "Q":
            return(-2)

        else:
            print("Zły input")


def monte_carlo_bj():
    l_count = 0
    w_count = 0
    d_count = 0
    for i in range(100000):
        wynik = bj()
        if wynik == 1:
            w_count += 1
        if wynik == 0:
            d_count += 1
        if wynik == -1:
            l_count += 1
    dzielnik = w_count + d_count + l_count
    print(w_count/dzielnik, d_count/dzielnik, l_count/dzielnik)
monte_carlo_bj()

def bj_full_game(start_budget):
    money_overall = start_budget
    print("Witaj w grze w blackjacka!!!")
    print("Akcje: H-dobranie, S-pass, Q-odejście do domu niczym przegryw (mama byłaby zawiedziona)...")
    print("")
    print("***************************************************************************")
    print("")
    while True:
        money = money_overall + 1
        if money_overall <= 0:
            print("BAKRUT!")
            break
        print(f"Obecny stan konta: {money_overall}.")
        while money > money_overall or money < 100:
            money = (input("Ile chcesz obstawić? Minimalna obstawialna kwota to 100: "))
            if money == "Q":
                print(f"Wracasz do domu z kwotą {money_overall}. Rodzice są zawiedzieni, żona odchodzi, dzieci płaczą.")
                return 0
            try:
                money = int(money)
            except:
                print("Musisz podać wartą liczbową lub symbol Q.")
        money_overall -= money
        wynik = bj_manual()
        if wynik == 1:
            money = 2*money
            print("Wygrana!")
        elif wynik == 0:
            money = money
            print("Remis!")
        elif wynik == -1:
            money = 0
            print("Przegrana!")
        elif wynik == -2:
            print(f"Wracasz do domu z kwotą {money_overall}. Rodzice są zawiedzieni, żona odchodzi, dzieci płaczą.")
            break
        money_overall += money
        print(" ")
        print("****************************************************  ")
        print(" ")
# bj_full_game(1000)