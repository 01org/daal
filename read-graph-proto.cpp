#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>

struct read_args_tag {};

class table {};

struct csv_read_ops_tag {};

class csv_data_source {
public:
    using tag_t = csv_read_ops_tag;
};

template <typename T, typename... Args>
using is_one_of = std::disjunction<std::is_same<T, Args>...>;

template <typename T, typename... Args>
constexpr bool is_one_of_v = is_one_of<T, Args...>::value;

template <class T, class U = void>
struct enable_if_type {
    using type = U;
};

template <typename T>
using enable_if_type_t = typename enable_if_type<T>::type;

template <typename T, typename Enable = void>
struct is_tagged : std::false_type {};

template <typename T>
struct is_tagged<T, enable_if_type_t<typename T::tag_t>> : std::true_type {};

template <typename T>
constexpr bool is_tagged_v = is_tagged<T>::value;

template <typename T, bool Enable = is_tagged_v<T>>
struct is_tag_one_of_impl {};

template <typename T>
struct is_tag_one_of_impl<T, true> {
    template <typename... Tags>
    static constexpr bool value = is_one_of_v<typename T::tag_t, Tags...>;
};

template <typename T>
struct is_tag_one_of_impl<T, false> {
    template <typename... Tags>
    static constexpr bool value = false;
};

template <typename T, typename... Tags>
struct is_tag_one_of {
    static constexpr bool value = is_tag_one_of_impl<T>::template value<Tags...>;
};

template <typename T, typename... Tags>
constexpr bool is_tag_one_of_v = is_tag_one_of<T, Tags...>::value;

template <typename Object, typename DataSource, typename Tag, typename... Options>
struct read_ops;

template <typename Object, typename DataSource, typename... Options>
using tagged_read_ops = read_ops<Object, DataSource, typename DataSource::tag_t, Options...>;

template <bool b, typename Tag, typename... Options>
struct read_ops_selector {};

template <typename Tag, typename Head, typename... Tail>
struct read_ops_selector<true, Tag, Head, Tail...> {
    typedef Head type;
};

template <typename T>
class has_tag_t_field {
private:
    typedef char yes_t[1];
    typedef char no_t[2];

    template <typename C>
    static yes_t& test(decltype(&C::tag_t));
    template <typename C>
    static no_t& test(...);

public:
    enum { value = (sizeof(test<T>(0)) == sizeof(yes_t)) };
};

template <typename Tag, typename Head, typename... Tail>
struct read_ops_selector<false, Tag, Head, Tail...> {
    typedef read_ops<table, csv_data_source, Head> type;
};

template <typename Object, typename DataSource, typename Head, typename... Tail>
auto read(const DataSource& ds, Head&& head, Tail&&... tail) {
    using head_t = std::decay_t<Head>;
    if constexpr (has_tag_t_field<head_t>::value) {
        using read_args_tag_t               = typename head_t::tag_t;
        constexpr bool is_valid_read_args_v = is_tag_one_of_v<read_args_tag_t, read_args_tag>;
        if constexpr (is_valid_read_args_v) {
            // Head is read_args
            using allocator_t = typename Head::allocator_t;
            using ops_t       = tagged_read_ops<Object, DataSource, allocator_t>;
            using args_t      = typename ops_t::args_t;
            return ops_t{}(ds, std::forward<Head>(head), std::forward<Tail>(tail)...);
            //  return ops_t{}(ds, std::forward<Head>(head));
        }
        else {
            using ops_t  = tagged_read_ops<Object, DataSource, Head>;
            using args_t = typename ops_t::args_t;
            return ops_t{}(ds, args_t{ std::forward<Head>(head), std::forward<Tail>(tail)... });
        }
    }
}

template <typename Object, typename DataSource>
auto read(const DataSource& ds) {
    using ops_t  = tagged_read_ops<Object, DataSource>;
    using args_t = typename ops_t::args_t;
    return ops_t{}(ds, args_t{});
}

struct default_allocator {};

// struct csv_read_args_tag {};

template <typename Object, typename DataSource, typename Allocator = default_allocator>
class csv_read_args {
public:
    using tag_t       = read_args_tag;
    using allocator_t = Allocator;
    csv_read_args(const Allocator& alloc) {}
    csv_read_args(const Allocator& alloc, int a) {}
    csv_read_args() {}
};

template <typename Object, typename DataSource, typename... Options>
struct csv_read_ops {
    using args_t   = csv_read_args<Object, DataSource, Options...>;
    using result_t = Object;

    result_t operator()(const DataSource& ds, const args_t& args) {
        return result_t{};
    }
};

template <typename Object, typename DataSource, typename... Options>
struct read_ops<Object, DataSource, csv_read_ops_tag, Options...>
        : csv_read_ops<Object, DataSource, Options...> {};

struct my_allocator {};

int main() {
    read<table>(csv_data_source{}, my_allocator{});
    read<table>(csv_data_source{}, my_allocator{}, 42);
    read<table>(csv_data_source{});
    my_allocator my_alloc;
    read<table>(csv_data_source{}, csv_read_args<table, csv_data_source, my_allocator>(my_alloc));
    return 0;
}