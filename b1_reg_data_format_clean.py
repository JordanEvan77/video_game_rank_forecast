# This will again take some reshaping. We probably want some or average of the key variables?
# Or something else that shows game to game improvement? We want to measure how someone can climb
# and improve, so maybe difference between the min and max, to show improvment? Or difference
# between average early on and average later on?


#I will need to convert the tier, rank and LP into a single integer, like a sliding scale,
# and merge that in from the all tier summoner ID lists I created. We don't have their initial rank,
# just their final rank. would be interesting to take wins and losses to get games played,
# and use that as an attribute too.

def convert_rank_to_int(tier, rank, lp):
    '''
    A way to turn this into an regression task
    :param tier:
    :param rank:
    :param lp:
    :return:
    '''
    tier_values = {'Iron': 1000, 'Bronze': 2000, 'Silver': 3000, 'Gold': 4000, 'Platinum': 5000,
        'Diamond': 6000}
    rank_values = {'IV': 100, 'III': 200, 'II': 300, 'I': 400 #TODO: isn't there a rank 5?
    }
    tier_value = tier_values[tier]
    rank_value = rank_values[rank]
    final_value = tier_value + rank_value + lp

    return final_value


final_rank = '' # will need to read in from original dataset and use funciton converter

columns = [
    "goldPerMinute",
    "damagePerMinute",
    "killParticipation",
    "laneMinionsFirst10Minutes",
    "champExperience",
    "baronKills",
    "dragonKills",
    "turretTakedowns",
    "inhibitorTakedowns",
    "visionScore",
    "controlWardsPlaced",
    "wardsKilled",
    "wardsPlaced",
    "damageSelfMitigated",
    "totalHeal",
    "takedowns",
    "deaths",
    "champExperience",
    "kills",
    "longestTimeSpentLiving",
    "riftHeraldTakedowns",
    "inhibitorKills",
    "totalDamageShieldedOnTeammates",
    "totalTimeSpentDead",
    "kda",
    "allInPings",
    "assistMePings",
    "commandPings",
    "dangerPings",
    "onMyWayPings",
    "basicPings",
    "pushPings",
    "visionClearedPings",
    "skillshotsHit",
    "neutralMinionsKilled"
]
