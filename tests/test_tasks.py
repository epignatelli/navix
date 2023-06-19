import jax
import navix as nx


def test_tasks_composition():
    reward_fn = nx.tasks.compose(
        nx.tasks.navigation,
        nx.tasks.action_cost,
        nx.tasks.time_cost,
        nx.tasks.wall_hit_cost,
    )

    env = nx.environments.Room(height=3, width=3, max_steps=8, reward_fn=reward_fn)
    key = jax.random.PRNGKey(0)

    def _test():
        timestep = env.reset(key)
        for _ in range(10):
            timestep = env.step(timestep, jax.random.randint(key, (), 0, 7))
        return timestep

    print(jax.jit(_test)())


if __name__ == "__main__":
    test_tasks_composition()
