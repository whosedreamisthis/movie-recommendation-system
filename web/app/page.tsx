import Image from "next/image";

async function getRecommendations(userId: string) {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/recommend?userId=${userId}`, {
    cache: 'no-store' // Ensures you get fresh data
  });

  if (!res.ok) throw new Error('Failed to fetch recommendations');
  return res.json();
}

export default async function Home({ searchParams }: { searchParams: Promise< { userId?: string }> }) {
  const {userId} = await searchParams; // Default to 1
  if (!userId) return <p>No user ID provided</p>;
  const movies = await getRecommendations(userId);
  console.log(userId);
  return (
      <main>
        <h1>Movie Recommendations</h1>
        <ul>
          {movies.map((movie: any) => (
              <li key={movie.movieId}>{movie.title}</li>
          ))}
        </ul>
      </main>);
}
