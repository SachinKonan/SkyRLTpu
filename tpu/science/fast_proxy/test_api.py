import unittest
import numpy as np
from . import Evaluator


def problem():
    return dict(initial_positions=np.array([[2., 2.], [5., 5.], [8., 8.]]),
                sizes=np.ones((3, 2)), fixed=np.array([True, False, False]), num_hard=1,
                canvas=np.array([10., 10.]), grid_shape=np.array([10, 10]),
                routes_per_micron=np.array([10., 10.]), macro_routing_allocation=np.array([1., 1.]),
                congestion_smoothing_range=2, port_positions=np.array([[0., 5.]]),
                pin_owner=np.array([0, 1, 2, 3]), pin_offset=np.zeros((4, 2)),
                net_offsets=np.array([0, 4]), net_weights=np.array([1.]), wirelength_normalizer=20.)


class ApiTests(unittest.TestCase):
    def test_transaction_and_rebuild(self):
        p = problem()
        with Evaluator(p) as ev:
            base = ev.score()
            ev.apply(1, [6, 5])
            with self.assertRaises(RuntimeError): ev.apply(2, [7, 8])
            with self.assertRaises(RuntimeError): ev.evaluate(p['initial_positions'])
            restored = ev.revert()
            np.testing.assert_allclose(list(base.values()), list(restored.values()), atol=1e-12)
            ev.apply(1, [6, 5]); ev.commit()
            changed = p['initial_positions'].copy(); changed[1] = [6, 5]
            np.testing.assert_allclose(list(ev.score().values()), list(ev.evaluate(changed).values()), atol=1e-12)

    def test_batch_does_not_change_state(self):
        p = problem()
        with Evaluator(p) as ev:
            base = ev.score()
            changed = p['initial_positions'].copy(); changed[1:] += .25
            rows = ev.evaluate_batch([changed, p['initial_positions']])
            self.assertEqual(len(rows), 2)
            np.testing.assert_allclose(list(ev.score().values()), list(base.values()), atol=1e-12)
            with Evaluator(p, changed) as other:
                np.testing.assert_allclose(list(rows[0].values()), list(other.score().values()), atol=1e-12)

    def test_move_batch_matches_full(self):
        p = problem()
        with Evaluator(p) as ev:
            rows = ev.evaluate_moves([1, 2, 1], [[6., 5.], [7., 8.], [4., 5.]])
            for (i, xy), row in zip([(1, [6, 5]), (2, [7, 8]), (1, [4, 5])], rows):
                pos = p['initial_positions'].copy(); pos[i] = xy
                np.testing.assert_allclose(list(row.values()), list(ev.evaluate(pos).values()), atol=1e-12)

    def test_hard_moves_and_periodic_rebuild(self):
        p = problem(); p['fixed'][0] = False
        with Evaluator(p) as ev:
            for step in range(260):
                xy = [2.0 + (step % 7) * .1, 2.0]
                ev.apply(0, xy); ev.commit()
                pos = p['initial_positions'].copy(); pos[0] = xy
                np.testing.assert_allclose(list(ev.score().values()), list(ev.evaluate(pos).values()), atol=1e-12)

    def test_bad_moves_leave_state_intact(self):
        with Evaluator(problem()) as ev:
            for i, xy in [(-1, [1, 1]), (3, [1, 1]), (0, [1, 1]), (1, [np.nan, 1]), (1, [11, 1])]:
                with self.assertRaises(ValueError): ev.apply(i, xy)
            ev.apply(1, [6, 5]); ev.revert()

    def test_bad_inputs(self):
        for field, value in [('pin_owner', np.array([0, 1, 2, 99])),
                             ('net_offsets', np.array([0, 5])),
                             ('routes_per_micron', [0, 0]),
                             ('grid_shape', [0, 1])]:
            p = problem(); p[field] = value
            with self.assertRaises(ValueError): Evaluator(p)

    def test_closed(self):
        ev = Evaluator(problem()); ev.close(); ev.close()
        with self.assertRaises(RuntimeError): ev.score()


if __name__ == '__main__': unittest.main()
