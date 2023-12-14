# pytorch

"""Tests for semirings."""

from absl.testing import absltest
import functools

import torch
import semirings


torch.set_printoptions(precision=8)  # precision for printing Tensors
assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def zero_and_one_test(semiring):
    torch.debug = True
    try:
        one = semiring.ones([3])
        zero = semiring.zeros([3])
        xs = torch.tensor([1.0, 2.0, 3.0])

        for args in [(one, xs), (xs, one)]:
            torch.testing.assert_close(semiring.times(*args), xs)
            torch.testing.assert_close(semiring.prod(torch.stack(args), dim=0), xs)

        for args in [(zero, xs), (xs, zero)]:
            torch.testing.assert_close(semiring.plus(*args), xs)
            print(
                f"zero_one_sum = {semiring.sum(torch.stack(args), dim=0)} @ {(semiring.sum(torch.stack(args), dim=0)).shape}"
            )
            torch.testing.assert_close(semiring.sum(torch.stack(args), dim=0), xs)

        torch.testing.assert_close(
            semiring.times(semiring.ones((1, 2)), semiring.zeros((3, 1))),
            semiring.zeros((3, 2)),
        )
        torch.testing.assert_close(
            semiring.times(semiring.zeros((1, 2)), semiring.ones((3, 1))),
            semiring.zeros((3, 2)),
        )
        torch.testing.assert_close(
            semiring.times(semiring.ones((1, 2)), semiring.ones((3, 1))),
            semiring.ones((3, 2)),
        )
        torch.testing.assert_close(
            semiring.times(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
            semiring.zeros((3, 2)),
        )

        torch.testing.assert_close(
            semiring.plus(semiring.ones((1, 2)), semiring.zeros((3, 1))),
            semiring.ones((3, 2)),
        )
        torch.testing.assert_close(
            semiring.plus(semiring.zeros((1, 2)), semiring.ones((3, 1))),
            semiring.ones((3, 2)),
        )
        torch.testing.assert_close(
            semiring.plus(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
            semiring.zeros((3, 2)),
        )

        torch.testing.assert_close(
            semiring.sum(torch.zeros([3, 0]), dim=0), torch.zeros([0])
        )
        torch.testing.assert_close(
            semiring.prod(torch.zeros([3, 0]), dim=0), torch.zeros([0])
        )

        torch.testing.assert_close(semiring.sum(torch.zeros([3, 0]), dim=1), zero)
        torch.testing.assert_close(semiring.prod(torch.zeros([3, 0]), dim=1), one)
    finally:
        torch.debug = False


def binary_op_broadcasting_test(semiring):
    def expected(op, x, y):
        expected_z = op(x, y)
        expected_z.backward(torch.ones_like(expected_z))
        expected_dx = x.grad
        expected_dy = y.grad
        return expected_z, expected_dx, expected_dy

    for op in [semiring.times, semiring.plus]:
        for shapes in [
            ([], [2]),
            ([1], [2]),
            ([1, 2], [3, 2]),
            ([2, 1], [2, 3]),
            ([3], [2, 3]),
        ]:
            for shape_x, shape_y in [shapes, shapes[::-1]]:
                err_msg = f"op={op} shapes={(shape_x, shape_y)}"
                x = torch.ones(shape_x, requires_grad=True)
                y = torch.ones(shape_y, requires_grad=True)
                z = op(x, y)
                z.backward(torch.ones_like(z))
                dx = x.grad
                dy = y.grad
                expected_z, expected_dx, expected_dy = expected(op, x, y)
                torch.testing.assert_close(z.detach(), expected_z.detach(), msg=err_msg)
                torch.testing.assert_close(
                    dx.detach(), expected_dx.detach(), msg=err_msg
                )
                torch.testing.assert_close(
                    dy.detach(), expected_dy.detach(), msg=err_msg
                )
                x.grad = None
                y.grad = None


class SemiringTest(absltest.TestCase):
    def test_value_shape(self):
        self.assertEqual(semirings.value_shape(torch.zeros([1, 2])), (1, 2))
        self.assertEqual(
            semirings.value_shape({"a": torch.zeros([1, 2]), "b": torch.ones([1, 2])}),
            (1, 2),
        )
        with self.assertRaisesRegex(
            ValueError, "No common shape can be derived for an empty PyTree"
        ):
            semirings.value_shape(None)
        with self.assertRaisesRegex(
            ValueError,
            "A semiring value must consist of ndarrays of a common shape",
        ):
            semirings.value_shape({"a": torch.zeros([1, 2]), "b": torch.ones([2])})


class RealTest(absltest.TestCase):
    def test_basics(self):
        torch.testing.assert_close(
            semirings.Real.times(torch.tensor(2), torch.tensor(3)), torch.tensor(6)
        )
        torch.testing.assert_close(
            semirings.Real.prod(torch.tensor([2, 3]), dim=0), torch.tensor(6)
        )
        torch.testing.assert_close(
            semirings.Real.plus(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )
        torch.testing.assert_close(
            semirings.Real.sum(torch.tensor([2, 3]), dim=0), torch.tensor(5)
        )
        zero_and_one_test(semirings.Real)
        binary_op_broadcasting_test(semirings.Real)


def check_sum_dim(self, semiring):
    """Checks that semiring sum handles dimensions correctly."""
    xs = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32, requires_grad=True).reshape(
        [2, 3, 4, 5]
    )
    xs.retain_grad()

    with self.subTest("forward"):
        self.assertEqual(semiring.sum(xs, dim=0).shape, torch.Size([3, 4, 5]))
        self.assertEqual(semiring.sum(xs, dim=1).shape, torch.Size([2, 4, 5]))
        self.assertEqual(semiring.sum(xs, dim=2).shape, torch.Size([2, 3, 5]))
        self.assertEqual(semiring.sum(xs, dim=3).shape, torch.Size([2, 3, 4]))
        self.assertEqual(semiring.sum(xs, dim=-1).shape, torch.Size([2, 3, 4]))
        self.assertEqual(semiring.sum(xs, dim=-2).shape, torch.Size([2, 3, 5]))
        self.assertEqual(semiring.sum(xs, dim=-3).shape, torch.Size([2, 4, 5]))
        self.assertEqual(semiring.sum(xs, dim=-4).shape, torch.Size([3, 4, 5]))
        with self.assertRaisesRegex(ValueError, "Invalid reduction dim"):
            semiring.sum(xs, dim=4)
        with self.assertRaisesRegex(ValueError, "Invalid reduction dim"):
            semiring.sum(xs, dim=-5)
        with self.assertRaisesRegex(ValueError, "Only int dim"):
            semiring.sum(xs, dim=None)  # type: ignore

    with self.subTest("backward"):

        def f(xs, dim):
            zs = semiring.sum(xs, dim=dim)
            while zs.shape:
                zs = torch.sum(zs, dim=0)
            zs.backward()
            return xs._grad

        for dim in range(-4, 4):
            self.assertEqual(f(xs, dim=dim).shape, xs.shape)


def check_sum_zero_sized(self, semiring):
    """Checks that semiring sum handles zero-sized dimensions correctly."""
    xs = torch.zeros([0, 2])

    torch.testing.assert_close(semiring.sum(xs, dim=0), semiring.zeros([2]))
    torch.testing.assert_close(semiring.sum(xs, dim=-2), semiring.zeros([2]))

    self.assertEqual(semiring.sum(xs, dim=1).shape, (0,))
    self.assertEqual(semiring.sum(xs, dim=-1).shape, (0,))


class LogTest(absltest.TestCase):
    def test_basics(self):
        torch.testing.assert_close(
            semirings.Log.times(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )
        self.assertEqual(
            semirings.Log.prod(torch.tensor([2, 3]), dim=0), torch.tensor(5)
        )
        torch.testing.assert_close(
            semirings.Log.plus(torch.tensor(2.0), torch.tensor(3.0)),
            torch.tensor(3.31326169),
        )
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([2, 3]), dim=0), torch.tensor(3.31326169)
        )
        zero_and_one_test(semirings.Log)
        binary_op_broadcasting_test(semirings.Log)

    def test_times_safety(self):
        inf = torch.tensor(float("inf"))
        self.assertTrue(torch.isnan(semirings.Log.times(-inf, inf)))
        self.assertTrue(torch.isnan(semirings.Log.times(inf, -inf)))
        torch.testing.assert_close(semirings.Log.times(inf, torch.tensor(1)), inf)
        torch.testing.assert_close(semirings.Log.times(torch.tensor(1), inf), inf)

    def test_prod_safety(self):
        inf = torch.tensor(float("inf"))
        self.assertTrue(
            torch.isnan(semirings.Log.prod(torch.tensor([-inf, inf]), dim=0))
        )
        self.assertTrue(
            torch.isnan(semirings.Log.prod(torch.tensor([inf, -inf]), dim=0))
        )
        torch.testing.assert_close(
            semirings.Log.prod(torch.tensor([inf, 1]), dim=0), inf
        )
        torch.testing.assert_close(
            semirings.Log.prod(torch.tensor([1, inf]), dim=0), inf
        )

    def test_plus_safety(self):
        inf = torch.tensor(float("inf"))
        torch.testing.assert_close(semirings.Log.plus(-inf, inf), inf)
        torch.testing.assert_close(semirings.Log.plus(inf, -inf), inf)
        torch.testing.assert_close(semirings.Log.plus(inf, torch.tensor(1)), inf)
        torch.testing.assert_close(semirings.Log.plus(torch.tensor(1), inf), inf)

    def test_sum_safety(self):
        inf = torch.tensor(float("inf"))
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([-inf, inf]), dim=0), inf
        )
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([inf, -inf]), dim=0), inf
        )
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([inf, 1]), dim=0), inf
        )
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([1, inf]), dim=0), inf
        )
        torch.testing.assert_close(
            semirings.Log.sum(torch.tensor([1, inf, -inf]), dim=0), inf
        )

    def test_log_plus_grad(self):
        inf = torch.tensor(float("inf"))

        def plus_grad(x, y):
            x = torch.tensor(x, requires_grad=True)
            y = torch.tensor(y, requires_grad=True)
            semirings.Log.plus(x, y).backward()
            return x._grad, y._grad

        for x, y, dx, dy in [
            (1.0, 1.0, 0.5, 0.5),
            (1.0, 2.0, 0.2689414213699951, 0.7310585786300049),
            (2.0, 1.0, 0.7310585786300049, 0.2689414213699951),
            (-inf, -inf, 0.0, 0.0),
            (1.0, -inf, 1.0, 0.0),
            (-inf, 1.0, 0.0, 1.0),
        ]:
            with self.subTest(f"x={x},y={y}"):
                # import pdb; pdb.set_trace()
                dx_, dy_ = plus_grad(x, y)
                torch.testing.assert_close(dx_, torch.tensor(dx))
                torch.testing.assert_close(dy_, torch.tensor(dy))
        for x, y in [(inf, inf), (1.0, inf), (inf, 1.0), (inf, -inf), (-inf, inf)]:
            with self.subTest(f"x={x},y={y}"):
                dx_, dy_ = plus_grad(x, y)
                if x == inf:
                    self.assertTrue(torch.isnan(dx_))
                else:
                    self.assertEqual(dx_, torch.tensor(0))
                if y == inf:
                    self.assertTrue(torch.isnan(dy_))
                else:
                    self.assertEqual(dy_, torch.tensor(0))
        with self.subTest("plus & times"):

            def f(x):
                x = torch.tensor(x, requires_grad=True)
                semirings.Log.plus(-inf, semirings.Log.times(-inf, x)).backward()
                return x._grad

            self.assertEqual(f(1.0), torch.tensor(0.0))

    def test_log_sum_grad(self):
        inf = torch.tensor(float("inf"))

        def sum_grad(xs):
            xs = torch.tensor(xs, requires_grad=True)
            torch.sum(semirings.Log.sum(xs, dim=0)).backward()
            return xs._grad

        for xs, dxs in [
            ([1.0, 1.0], [0.5, 0.5]),
            ([1.0, 2.0], [0.2689414213699951, 0.7310585786300049]),
            ([2.0, 1.0], [0.7310585786300049, 0.2689414213699951]),
            ([-inf, -inf], [0.0, 0.0]),
            ([1.0, -inf], [1.0, 0.0]),
            ([-inf, 1.0], [0.0, 1.0]),
            ([-inf, 1.0, 2.0], [0, 0.2689414213699951, 0.7310585786300049]),
        ]:
            with self.subTest(f"xs={xs}"):
                dxs_ = sum_grad(xs)
                torch.testing.assert_close(dxs_, torch.tensor(dxs))
        for xs in [
            (inf, inf),
            (1, inf),
            (inf, 1),
            (inf, -inf),
            (-inf, inf),
            (inf, -inf, 1),
        ]:
            with self.subTest(f"xs={xs}"):
                dxs_ = sum_grad(xs)
                xs = torch.tensor(xs)
                torch.testing.assert_close(torch.isnan(dxs_), xs == inf)
                # # TODO: SKIPPED TEST
                # torch.testing.assert_close(
                #     torch.where(xs != inf, dxs_, torch.tensor(0)), torch.tensor([0])
                # )

        with self.subTest("sum & prod"):

            def f(x):
                x = torch.tensor(x, requires_grad=True)
                semirings.Log.sum(
                    torch.stack(
                        [
                            -inf,
                            semirings.Log.prod(
                                torch.stack([-inf, x]),
                                dim=0,
                            ),
                        ]
                    ),
                    dim=0,
                ).backward()
                return x._grad

            self.assertEqual(f(1.0), torch.tensor(0.0))

    def test_log_sum_dim(self):
        check_sum_dim(self, semirings.Log)

    def test_log_sum_zero_sized(self):
        check_sum_zero_sized(self, semirings.Log)


class MaxTropicalTest(absltest.TestCase):
    def test_basics(self):
        torch.testing.assert_close(
            semirings.MaxTropical.times(torch.tensor(2), torch.tensor(3)),
            torch.tensor(5),
        )
        torch.testing.assert_close(
            semirings.MaxTropical.prod(torch.tensor([2, 3]), dim=0), torch.tensor(5)
        )
        torch.testing.assert_close(
            semirings.MaxTropical.plus(torch.tensor(2), torch.tensor(3)),
            torch.tensor(3),
        )
        torch.testing.assert_close(
            semirings.MaxTropical.sum(torch.tensor([2, 3]), dim=0), torch.tensor([3])
        )
        zero_and_one_test(semirings.MaxTropical)
        binary_op_broadcasting_test(semirings.MaxTropical)

    def test_plus_grad(self):
        def plus_grad(a, b):
            a = torch.tensor(a, requires_grad=True)
            b = torch.tensor(b, requires_grad=True)
            torch.sum(semirings.MaxTropical.plus(a, b)).backward()
            return a._grad, b._grad

        x = [[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]]
        torch.testing.assert_close(
            (plus_grad(x[0], x[1])),
            (torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        )

    def test_sum_grad(self):
        def sum_grad(x, dim):
            x = torch.tensor(x, requires_grad=True)
            torch.sum(semirings.MaxTropical.sum(x, dim=dim)).backward()
            return x._grad

        xs = [[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]]
        yt = (torch.tensor([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])).T

        torch.testing.assert_close(
            torch.sum(sum_grad(xs, dim=0)),
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            torch.sum(sum_grad(yt, dim=1)),
            (torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])).T,
        )

    def test_sum_dim(self):
        check_sum_dim(self, semirings.MaxTropical)

    def test_sum_zero_sized(self):
        check_sum_zero_sized(self, semirings.MaxTropical)


if __name__ == "__main__":
    absltest.main()
