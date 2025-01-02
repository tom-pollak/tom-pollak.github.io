# Building an Automated Horse Trading System -- Each Way Matcher

> https://github.com/tom-pollak/each-way-matcher
> This is quite an old project, but I just got around to publishing all the old notes I had on it

During Covid, I did a lot of match betting, which involves exploiting bookmakers "free bet" offers to attract gamblers. After profiting from most of the easy offers, I started looking into new strategies and discovered an edge with two strategies in a type of hose racing bet called each-way.

> Each-way: Your bet is split in two -- half the stake goes to the horse winning, half goes towards the horse "placing" (coming 2nd or 3rd). The kicker: place payout is calculated by `win_odds / 5`.

This isn't independant, so if a race has a huge favourite to win (e.g. 1.05 odds), then all other horses will be given high win odds, but at least *some* horses must come 2nd or 3rd. These horses may have underated place value relative to their high win odds. Three tasks:

1. Find undervalued horses by calculating probability of horse finishing in *any arbitary place* based on all horses win odds. [Github Gist](https://gist.github.com/tom-pollak/818f2d1dd21575bc7924bc44feb1d50d)

2. Use an adapted 3 way Kelly Criterion for optimal bet sizing (win, place, lose) and geometric mean growth rate for ranking bets (I kind of rediscovered in the frustration of Expected Value not taking into account a finite bankroll / risk or ruin).

3. Write an automated script to scrape odds and place bets via Selenium on the Bookmaker, while laying (eqiv shorting) on the Betfair API. Also had to recover from non matched bets, which involved just "backing off" if the odds changed (compute new bets to maximize Kelly).

4. Run on a headless Raspberry Pi

5. Profit.

## Second strategy: Extra place matching

It's surprising how deep you can go on a topic once you've started digging. When building this first betting strategy I found an interesting offer given for each way. Bookmakers offering an "extra place" for their each way bets.

> Extra place matching. Bookmakers will pay out place bets for an extra place: 2nd, 3rd and **4th**.

Why is this interesting? Well apart from the slightly better statisitcal edge, the exchange *doesn't pay out the extra place*. So that means we can make a "match bet" with the bookmaker and exchange, which will make a small loss because of spread and commission etc. However if the horse comes 4th, we win the "extra-place" bet with the bookmaker *and* the lay bet on the exchange for a large profit.

- I could predict the odds of a horse coming 4th by its win oddds (see gist above) I can get the exact EV for any given horse and race.
- I used the same Kelly technique for optimal bet sizing, now with four outcomes. (Win Place, Extra Place, Loss).

This was turning out to be quite profitable, actually better than my first method, which was more akin to a statistical arb.

## Getting Banned

It was a long time coming, but eventually I was rate-limited by the bookmaker which essentially shut down the operation.

I could have continued with new bookmakers but decided against it since a) Most had already banned me for manual match betting. b) The small niche that allowed me to easily exploit the market also meant it had small volume, so I wouldn't be able to scale much.

So I chalked it up to a good experience and moved on. At its peak I had thousands in liability locked in automated trades. 
